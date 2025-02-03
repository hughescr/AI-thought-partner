import { describe, it, expect, beforeEach, afterEach } from 'bun:test';
import { NovelDocumentStore, NovelDocument, computeNovelID } from '../lib/NovelDocumentStore';
import { ChapterDocumentStore } from '../lib/ChapterDocumentStore';
import fs from 'fs/promises';
import path from 'path';
import _ from 'lodash';

import { RunnableLambda } from '@langchain/core/runnables';
import { ChapterSummaryGenerator } from '../lib/ChapterSummaryGenerator';
import Loki from 'lokijs';

let chapterStore: ChapterDocumentStore;

const TEST_DB_PATH = path.join(import.meta.dir, 'test-novels.db');

describe('NovelDocumentStore', () => {
    let novelStore: NovelDocumentStore;

    beforeEach(async () => {
        try {
            await fs.unlink(TEST_DB_PATH);
        } catch{ /* ignore error */ }
        const chaptersDb = new Loki('chapters_test.db', {
            adapter: new Loki.LokiFsAdapter(),
            autoload: true,
            autosave: true,
            autosaveInterval: 5000,
        });
        const summaryGenerator = new ChapterSummaryGenerator({
            llm: RunnableLambda.from(_.constant('Concise generated summary')),
            targetSummarySize: 100,
        });
        chapterStore = new ChapterDocumentStore(chaptersDb, summaryGenerator);
        // Construct novelStore with a given file path and the real chapter store.
        novelStore = new NovelDocumentStore(TEST_DB_PATH, chapterStore);
    });

    afterEach(async () => {
        try {
            await fs.unlink(TEST_DB_PATH);
        } catch{ /* ignore error */ }
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
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- accessing private collection for verification in tests only.
        const coll = (novelStore as any).db.getCollection('novels');
        expect(coll.findOne({ 'metadata.novelID': providedID })).toBeDefined();
        // eslint-disable-next-line lodash/prefer-lodash-method -- collection is not an array
        const chaptersAdded = (chapterStore as unknown as { collection: Loki.Collection }).collection.find();
        expect(chaptersAdded.length).toBe(1);
        const chapter = chaptersAdded[0];
        expect(chapter.metadata.chapter).toBe(1);
        expect(chapter.metadata.novelID).toBe(providedID);
    });
});
afterAll(async () => {
    try {
        await fs.unlink('chapters_test.db');
    } catch { /* ignore error */ }
    try {
        await fs.unlink('test-chunks.db');
    } catch { /* ignore error */ }
});

describe('computeNovelID', () => {
    it('computes a consistent novelID', () => {
        expect(computeNovelID(' John Doe ', ' Test Novel ')).toBe('john_doe_test_novel');
    });
});
