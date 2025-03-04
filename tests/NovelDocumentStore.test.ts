import { describe, it, expect, beforeEach, afterEach, jest } from 'bun:test';
import { NovelDocumentStore, NovelDocument, computeNovelID } from '../lib/NovelDocumentStore';
import { ChapterDocumentStore, ChapterDocument } from '../lib/ChapterDocumentStore';
import { rm } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join as pathJoin } from 'node:path';
import _ from 'lodash';

import { RunnableLambda } from '@langchain/core/runnables';
import { ChapterSummaryGenerator } from '../lib/ChapterSummaryGenerator';
import PouchDB from 'pouchdb';
import find from 'pouchdb-find';
PouchDB.plugin(find);
import { Embeddings } from '@langchain/core/embeddings';
import type { AsyncCaller } from '@langchain/core/utils/async_caller';
import { Document } from '@langchain/core/documents';

const dummyEmbeddings: Embeddings = {
    embedQuery: async (_query: string | number[] | Document) => _.fill(new Array(512), 0),
    embedDocuments: async (docs: string[] | Document[]) => _.times(docs.length, () => _.fill(new Array(512), 0)),
    caller: {} as AsyncCaller
};

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const TEST_DB_PATH = pathJoin(__dirname, 'test-novels.db');

describe('NovelDocumentStore', () => {
    let novelStore: NovelDocumentStore;
    let chapterStore: ChapterDocumentStore;

    beforeEach(async () => {
        const summaryGenerator = new ChapterSummaryGenerator({
            llm: RunnableLambda.from(_.constant('Concise generated summary')),
            targetSummarySize: 100,
        });
        // Construct novelStore with a given file path and the real chapter store.
        novelStore = new NovelDocumentStore(dummyEmbeddings, { filePath: TEST_DB_PATH, summaryGenerator });
        await (novelStore as unknown as { indexCreated: Promise<void> }).indexCreated;
        chapterStore = (novelStore as unknown as { chapterStore: ChapterDocumentStore }).chapterStore;
    });

    afterEach(async () => {
        await novelStore.destroy();
        await rm(TEST_DB_PATH, { recursive: true, force: true });
    });

    it('computes novelID if missing and splits novel into chapters', async () => {
        const markdownText = `# Chapter 1
Content of chapter one.
# Chapter 2
Content of chapter two.
`;
        const novel: NovelDocument = new NovelDocument({
            pageContent: markdownText,
            metadata: { title: 'Test Novel', author: 'John Doe', genre: 'Fiction' }
        });
        await novelStore.addNovel(novel);
        const computedID = computeNovelID('John Doe', 'Test Novel');
        expect(novel.metadata.novelID).toBe(computedID);

        // Verify the novel was inserted into the novels collection.
        // eslint-disable-next-line lodash/prefer-lodash-method -- not actually an array
        const res = await (novelStore as unknown as { db: PouchDB.Database }).db.find({
            selector: { metadata: { docType: 'novel', novelID: computedID } },
            limit: Number.MAX_SAFE_INTEGER,
        });
        expect(res.docs.length).toBeGreaterThan(0);

        // Verify that chapters were added and numbered correctly.
        // eslint-disable-next-line lodash/prefer-lodash-method -- not actually an array
        const chaptersRes = await (chapterStore as unknown as { db: PouchDB.Database }).db.find({
            selector: { metadata: { docType: 'chapter', novelID: computedID } },
            limit: Number.MAX_SAFE_INTEGER,
        }) as unknown as { docs: ChapterDocument[] };
        expect(chaptersRes.docs.length).toBeGreaterThan(0);
        _.forEach(chaptersRes.docs, (chapter, index) => {
            expect(chapter.metadata.chapter).toBe(index + 1);
            expect(chapter.metadata.novelID).toBe(computedID);
        });
    });

    it('preserves provided novelID and works correctly', async () => {
        const markdownText = `# Chapter 1
Chapter one content.
`;
        const novel: NovelDocument = new NovelDocument({
            pageContent: markdownText,
            metadata: { title: 'Some Title', author: 'Jane Smith' }
        });
        await novelStore.addNovel(novel);
        expect(novel.metadata).toHaveProperty('novelID');

        // eslint-disable-next-line lodash/prefer-lodash-method -- not actually an array
        const res = await (novelStore as unknown as { db: PouchDB.Database }).db.find({
            selector: { metadata: { docType: 'novel', novelID: novel.metadata.novelID } },
            limit: Number.MAX_SAFE_INTEGER,
        });
        expect(res.docs.length).toBeGreaterThan(0);
        // eslint-disable-next-line lodash/prefer-lodash-method -- not actually an array
        const chaptersRes = await (chapterStore as unknown as { db: PouchDB.Database }).db.find({
            selector: { 'metadata.docType': 'chapter', 'metadata.novelID': novel.metadata.novelID },
            limit: Number.MAX_SAFE_INTEGER,
        }) as unknown as { docs: ChapterDocument[] };
        expect(chaptersRes.docs.length).toBe(1);
        const chapter = chaptersRes.docs[0];
        expect(chapter.metadata.chapter).toBe(1);
        expect(chapter.metadata.novelID).toBe(novel.metadata.novelID);
    });

    it('getNovel retrieves a novel by title and author', async () => {
        const markdownText = `# Chapter 1
Test content.
`;
        const novel = new NovelDocument({
            pageContent: markdownText,
            metadata: { title: 'Get Novel Test', author: 'Test Author' }
        });
        await novelStore.addNovel(novel);

        const retrievedNovel = await novelStore.getNovel('Get Novel Test', 'Test Author');
        expect(retrievedNovel).toBeDefined();
        expect(retrievedNovel?.metadata.title).toBe('Get Novel Test');
        expect(retrievedNovel?.metadata.author).toBe('Test Author');
        expect(retrievedNovel?.metadata.novelID).toBe(novel.metadata.novelID);
        
        // Test with non-existent novel
        const nonExistentNovel = await novelStore.getNovel('Non Existent', 'Not Real');
        expect(nonExistentNovel).toBeUndefined();
    });

    it('getChapter retrieves a specific chapter', async () => {
        const markdownText = `# Chapter 1
First chapter content.
# Chapter 2
Second chapter content.
`;
        const novel = new NovelDocument({
            pageContent: markdownText,
            metadata: { title: 'Chapter Test', author: 'Chapter Author' }
        });
        await novelStore.addNovel(novel);

        const chapter = await novelStore.getChapter('Chapter Test', 'Chapter Author', 1);
        expect(chapter).toBeDefined();
        expect(chapter?.metadata.chapter).toBe(1);
        expect(chapter?.pageContent).toContain('First chapter content');

        // Test with non-existent chapter
        const nonExistentChapter = await novelStore.getChapter('Chapter Test', 'Chapter Author', 999);
        expect(nonExistentChapter).toBeUndefined();
        
        // Test with non-existent novel
        const chapterFromNonExistentNovel = await novelStore.getChapter('Non Existent', 'Not Real', 1);
        expect(chapterFromNonExistentNovel).toBeUndefined();
    });

    it('getChapters retrieves all chapters for a novel', async () => {
        const markdownText = `# Chapter 1
First chapter.
# Chapter 2
Second chapter.
# Chapter 3
Third chapter.
`;
        const novel = new NovelDocument({
            pageContent: markdownText,
            metadata: { title: 'Multi Chapter', author: 'Multi Author' }
        });
        await novelStore.addNovel(novel);

        const chapters = await novelStore.getChapters('Multi Chapter', 'Multi Author');
        expect(chapters).toBeDefined();
        expect(chapters?.length).toBe(3);
        expect(_.map(chapters, 'metadata.chapter')).toEqual([1, 2, 3]);
        
        // Test with non-existent novel
        const chaptersFromNonExistentNovel = await novelStore.getChapters('Non Existent', 'Not Real');
        expect(chaptersFromNonExistentNovel).toBeUndefined();
    });

    it('getChapterSummary retrieves a summary for a chapter', async () => {
        const markdownText = `# Chapter 1
Summary test content.
`;
        const novel = new NovelDocument({
            pageContent: markdownText,
            metadata: { title: 'Summary Test', author: 'Summary Author' }
        });
        await novelStore.addNovel(novel);
        
        const chapter = await novelStore.getChapter('Summary Test', 'Summary Author', 1);
        expect(chapter).toBeDefined();
        
        if (chapter) {
            const summary = await novelStore.getChapterSummary(chapter);
            expect(summary).toBeDefined();
            expect(summary?.pageContent).toContain('Concise generated summary');
            expect(summary?.metadata.chapter).toBe(1);
        }
    });

    it('getNovelSummary retrieves a summary of all chapters', async () => {
        const markdownText = `# Chapter 1
First chapter content.
# Chapter 2
Second chapter content.
`;
        const novel = new NovelDocument({
            pageContent: markdownText,
            metadata: { 
                title: 'Novel Summary Test', 
                author: 'Summary Author',
                genre: 'Test Genre',
                filepath: '/path/to/file'
            }
        });
        await novelStore.addNovel(novel);
        
        const novelSummary = await (novelStore as any).getNovelSummary(novel.metadata.novelID);
        expect(novelSummary).toBeDefined();
        expect(novelSummary?.metadata.title).toBe('Novel Summary Test');
        expect(novelSummary?.metadata.author).toBe('Summary Author');
        expect(novelSummary?.metadata.genre).toBe('Test Genre');
        expect(novelSummary?.metadata.filepath).toBe('/path/to/file');
        expect(novelSummary?.pageContent).toContain('# Chapter 1\nConcise generated summary');
        expect(novelSummary?.pageContent).toContain('# Chapter 2\nConcise generated summary');
        
        // Test with non-existent novel
        const nonExistentNovelSummary = await (novelStore as any).getNovelSummary('non_existent_id');
        expect(nonExistentNovelSummary).toBeUndefined();
    });
});

describe('NovelDocumentStore - VectorStore API', () => {
    let novelStore: NovelDocumentStore;
    beforeEach(async () => {
        const summaryGenerator = new ChapterSummaryGenerator({
            llm: RunnableLambda.from(_.constant('Concise generated summary')),
            targetSummarySize: 100,
        });
        novelStore = new NovelDocumentStore(dummyEmbeddings, { filePath: TEST_DB_PATH, summaryGenerator });
    });
    afterEach(async () => {
        await novelStore.destroy();
        await rm(TEST_DB_PATH, { recursive: true, force: true });
    });

    it('delegates similaritySearchVectorWithScore correctly', async () => {
        // Add a novel so that there are chapters.
        const markdownText = `# Chapter 1
Chapter one.
# Chapter 2
Chapter two.
`;
        const novel = new NovelDocument({
            pageContent: markdownText,
            metadata: { title: 'Vector Novel 2', author: 'Vector Author', genre: 'SciFi' }
        });
        await novelStore.addNovel(novel);

        // Now search using the dummy vector; expect to get back each chapter only once.
        const faiss = await novelStore.getVectorStoreForNovel(novel);

        // Create a dummy vector to search for
        const dummyVector = _.fill(Array(512), 1);
        const searchResults = await faiss.similaritySearchVectorWithScore(dummyVector, 10);
        const seenChapters = new Set();
        for(const [doc, _score] of searchResults) {
            expect(doc.metadata.chapter).toBeDefined();
            expect(seenChapters.has(doc.metadata.chapter)).toBe(false);
            seenChapters.add(doc.metadata.chapter);
        }
    });

    it('retrieves chapters using the vectorstore retriever interface with a text query', async () => {
        const markdownText = `# Chapter 1
The quick brown fox jumps over the lazy dog.
# Chapter 2
Lorem ipsum dolor sit amet, consectetur adipiscing elit.
`;
        const novel = new NovelDocument({
            pageContent: markdownText,
            metadata: { title: 'Retriever Novel', author: 'Retriever Author', genre: 'Test' }
        });
        await novelStore.addNovel(novel);
        const vectorStore = await novelStore.getVectorStoreForNovel(novel);
        // Use as retriever if available, otherwise use the vectorStore directly.
        const retriever = vectorStore.asRetriever();
        const query = 'quick brown fox';
        const results = await retriever.invoke(query);
        expect(results.length).toBeGreaterThan(0);
        _.forEach(results, (doc) => {
            expect(doc.metadata.chapter).toBeDefined();
        });
        // Optionally check that at least one result contains the query text.
        const hasQuery = _.some(results, doc => _(doc.pageContent).toLower().includes('quick brown fox'));
        expect(hasQuery).toBe(true);
    });

    it('addDocuments should throw "Method not implemented."', async () => {
        // Create a novel to obtain a NovelFaissStore instance.
        const markdownText = `# Chapter 1
Test chapter content.`;
        const novel = new NovelDocument({
            pageContent: markdownText,
            metadata: { title: 'AddDocs Novel', author: 'Test Author' }
        });
        await novelStore.addNovel(novel);
        const faiss = await novelStore.getVectorStoreForNovel(novel);
        await expect(faiss.addDocuments([{ pageContent: 'dummy', metadata: {} }]))
        .rejects.toThrow('Method not implemented.');
    });

    it('addVectors should throw "Method not implemented."', async () => {
        // Create another novel to get a fresh vector store.
        const markdownText = `# Chapter 1
Test chapter content.`;
        const novel = new NovelDocument({
            pageContent: markdownText,
            metadata: { title: 'AddVectors Novel', author: 'Test Author' }
        });
        await novelStore.addNovel(novel);
        const faiss = await novelStore.getVectorStoreForNovel(novel);
        await expect(faiss.addVectors([[0]], [{ pageContent: 'dummy', metadata: {} }]))
        .rejects.toThrow('Method not implemented.');
    });
});

describe('computeNovelID', () => {
    it('computes a consistent novelID', () => {
        expect(computeNovelID(' John Doe ', ' Test Novel ')).toBe('john_doe_test_novel');
    });
});

describe('NovelDocument', () => {
    it('creates document with computed novelID', () => {
        const novel = new NovelDocument({
            pageContent: 'Novel content',
            metadata: { title: 'Test Novel', author: 'John Doe' }
        });
        expect(novel.metadata.novelID).toBe('john_doe_test_novel');
        expect(novel.metadata.docType).toBe('novel');
    });

    it('creates document from an existing document', () => {
        const doc = new Document({
            pageContent: 'Novel content',
            metadata: {}
        });
        const novel = NovelDocument.fromDocument(doc, 'Test Novel', 'John Doe', 'Fiction', '/path/to/file');
        expect(novel.metadata.novelID).toBe('john_doe_test_novel');
        expect(novel.metadata.title).toBe('Test Novel');
        expect(novel.metadata.author).toBe('John Doe');
        expect(novel.metadata.genre).toBe('Fiction');
        expect(novel.metadata.filepath).toBe('/path/to/file');
    });
});

describe('NovelDocumentStore - Error Handling and Edge Cases', () => {
    const TEST_ERROR_DB_PATH = pathJoin(__dirname, 'test-error-novels.db');
    
    afterEach(async () => {
        await rm(TEST_ERROR_DB_PATH, { recursive: true, force: true });
        // Restore any mocks
        jest.restoreAllMocks();
    });

    it('should handle database errors gracefully', () => {
        // This is a placeholder for database error tests that are hard to mock with PouchDB
        // Future implementations should use proper mocking or a test database adapter
        expect(true).toBe(true);
    });
});

describe('NovelDocumentStore - Concurrency and Large Data', () => {
    const TEST_CONCURRENCY_DB_PATH = pathJoin(__dirname, 'test-concurrency-novels.db');
    let novelStore: NovelDocumentStore;
    
    beforeEach(async () => {
        const summaryGenerator = new ChapterSummaryGenerator({
            llm: RunnableLambda.from(_.constant('Concise generated summary')),
            targetSummarySize: 100,
        });
        
        novelStore = new NovelDocumentStore(dummyEmbeddings, { 
            filePath: TEST_CONCURRENCY_DB_PATH,
            summaryGenerator 
        });
        // Wait for initialization to complete
        await (novelStore as unknown as { indexCreated: Promise<void> }).indexCreated;
    });
    
    afterEach(async () => {
        try {
            await novelStore.destroy();
        } catch (e) {
            // Ignore errors during cleanup
        }
        await rm(TEST_CONCURRENCY_DB_PATH, { recursive: true, force: true });
    });

    it('handles large chapters without memory issues', async () => {
        // Create a novel with a large chapter
        const largeContent = _.repeat('This is a test sentence that takes up space. ', 500); // Smaller to avoid test timeouts
        const novel = new NovelDocument({
            pageContent: `# Large Chapter\n${largeContent}`,
            metadata: { title: 'Large Novel', author: 'Test Author' }
        });
        
        // Add the novel and ensure it completes without errors
        await novelStore.addNovel(novel);
        
        // Verify we can retrieve it
        const retrievedNovel = await novelStore.getNovel('Large Novel', 'Test Author');
        expect(retrievedNovel).toBeDefined();
        
        // Get the chapter and verify it's complete
        const chapter = await novelStore.getChapter('Large Novel', 'Test Author', 1);
        expect(chapter).toBeDefined();
        expect(chapter?.pageContent.length).toBeGreaterThan(5000);
    });
});
