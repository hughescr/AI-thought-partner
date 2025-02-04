import { describe, it, expect, beforeEach, afterEach } from 'bun:test';
import { NovelDocumentStore, NovelDocument, computeNovelID } from '../lib/NovelDocumentStore';
import { ChapterDocumentStore } from '../lib/ChapterDocumentStore';
import { unlink, access, rm } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join as pathJoin } from 'node:path';
import _ from 'lodash';

import { RunnableLambda } from '@langchain/core/runnables';
import { ChapterSummaryGenerator } from '../lib/ChapterSummaryGenerator';
import Loki from 'lokijs';
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
        try {
            await access(TEST_DB_PATH);
            throw new Error(`Test database file ${TEST_DB_PATH} already exists. Aborting.`);
        } catch{ /* file does not exist; continue */ }
        const summaryGenerator = new ChapterSummaryGenerator({
            llm: RunnableLambda.from(_.constant('Concise generated summary')),
            targetSummarySize: 100,
        });
        // Construct novelStore with a given file path and the real chapter store.
        novelStore = new NovelDocumentStore(dummyEmbeddings, { filePath: TEST_DB_PATH, summaryGenerator });
        chapterStore = (novelStore as unknown as { chapterStore: ChapterDocumentStore }).chapterStore;
    });

    afterEach(async () => {
        await chapterStore.close();
        await novelStore.close();
        try {
            await unlink(TEST_DB_PATH);
        } catch{ /* ignore error */ }
        try {
            await rm(TEST_DB_PATH + '-FAISS', { recursive: true, force: true });
        } catch{
            // Ignore errors if the directory does not exist.
        }
    });

    it('computes novelID if missing and splits novel into chapters', async () => {
        const markdownText = `# Chapter 1
Content of chapter one.
# Chapter 2
Content of chapter two.
`;
        const novel: NovelDocument = new NovelDocument({
            pageContent: markdownText,
            metadata: { novelID: '', title: 'Test Novel', author: 'John Doe', genre: 'Fiction' }
        });
        await novelStore.addNovel(novel);
        const computedID = computeNovelID('John Doe', 'Test Novel');
        expect(novel.metadata.novelID).toBe(computedID);
        // Verify the novel was inserted into the novels collection.
        const coll = (novelStore as unknown as { db: Loki }).db.getCollection('novels');
        expect(coll.findOne({ 'metadata.novelID': computedID })).toBeDefined();
        // Verify that chapters were added and numbered correctly.
        // eslint-disable-next-line lodash/prefer-lodash-method -- collection is not an array
        const chaptersAdded = (chapterStore as unknown as { collection: Loki.Collection }).collection.find();
        expect(chaptersAdded.length).toBeGreaterThan(0);
        _.forEach(chaptersAdded, (chapter, index) => {
            expect(chapter.metadata.chapter).toBe(index + 1);
            expect(chapter.metadata.novelID).toBe(computedID);
        });
    });

    it('preserves provided novelID and works correctly', async () => {
        const providedID = 'custom_novel';
        const markdownText = `# Chapter 1
Chapter one content.
`;
        const novel: NovelDocument = new NovelDocument({
            pageContent: markdownText,
            metadata: { novelID: providedID, title: 'Some Title', author: 'Jane Smith' }
        });
        await novelStore.addNovel(novel);
        expect(novel.metadata.novelID).toBe(providedID);

        const coll = (novelStore as unknown as { db: Loki }).db.getCollection('novels');
        expect(coll.findOne({ 'metadata.novelID': providedID })).toBeDefined();
        // eslint-disable-next-line lodash/prefer-lodash-method -- collection is not an array
        const chaptersAdded = (chapterStore as unknown as { collection: Loki.Collection }).collection.find();
        expect(chaptersAdded.length).toBe(1);
        const chapter = chaptersAdded[0];
        expect(chapter.metadata.chapter).toBe(1);
        expect(chapter.metadata.novelID).toBe(providedID);
    });
});

describe('NovelDocumentStore - VectorStore API', () => {
    let novelStore: NovelDocumentStore;
    let chapterStore: ChapterDocumentStore;
    beforeEach(async () => {
        try {
            await access(TEST_DB_PATH);
            throw new Error(`Test database file ${TEST_DB_PATH} already exists. Aborting.`);
        } catch{ /* file does not exist; continue */ }
        const summaryGenerator = new ChapterSummaryGenerator({
            llm: RunnableLambda.from(_.constant('Concise generated summary')),
            targetSummarySize: 100,
        });
        novelStore = new NovelDocumentStore(dummyEmbeddings, { filePath: TEST_DB_PATH, summaryGenerator });
        chapterStore = (novelStore as unknown as { chapterStore: ChapterDocumentStore }).chapterStore;
    });
    afterEach(async () => {
        await chapterStore.close();
        await novelStore.close();
        try {
            await unlink(TEST_DB_PATH);
        } catch{ /* ignore error */ }
    });
    it('returns _vectorstoreType as "novel"', () => {
        expect(novelStore._vectorstoreType()).toBe('novel');
    });

    it('can add documents via addDocuments (calling addNovel internally)', async () => {
        const markdownText = `# Chapter 1
Chapter one content.
# Chapter 2
Chapter two content.
`;
        const novel = new NovelDocument({
            pageContent: markdownText,
            metadata: { title: 'Vector Novel', author: 'Vector Author', genre: 'SciFi' }
        });
        await novelStore.addDocuments([novel]);
        // The novelID should be computed.
        const computedID = computeNovelID(novel.metadata.author, novel.metadata.title);
        // Verify the novels collection has the inserted novel.
        const coll = (novelStore as unknown as { db: Loki }).db.getCollection('novels');
        expect(coll.findOne({ 'metadata.novelID': computedID })).toBeDefined();
        // Verify that chapters have been created.
        // eslint-disable-next-line lodash/prefer-lodash-method -- collection is not an array
        const chapters = (chapterStore as unknown as { collection: Loki.Collection }).collection.find();
        expect(chapters.length).toBeGreaterThan(1);
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
        // Create a dummy vector for each chapter (e.g. an array of 512 ones).
        const dummyVector = _.fill(Array(512), 1);

        // Now search using the dummy vector; expect to get back each chapter only once.
        const searchResults = await novelStore.similaritySearchVectorWithScore(dummyVector, 10);
        const seenChapters = new Set();
        for(const [doc, _score] of searchResults) {
            expect(doc.metadata.chapter).toBeDefined();
            expect(seenChapters.has(doc.metadata.chapter)).toBe(false);
            seenChapters.add(doc.metadata.chapter);
        }
    });
});

describe('computeNovelID', () => {
    it('computes a consistent novelID', () => {
        expect(computeNovelID(' John Doe ', ' Test Novel ')).toBe('john_doe_test_novel');
    });
});
