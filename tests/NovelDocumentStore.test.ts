import { describe, it, expect, beforeEach, afterEach } from 'bun:test';
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
