import { describe, it, expect, beforeEach, afterEach } from 'bun:test';
import { NovelDocumentStore, NovelDocument, computeNovelID } from '../lib/NovelDocumentStore';
import { ChapterDocument, ChapterDocumentStore } from '../lib/ChapterDocumentStore';
import fs from 'fs/promises';
import path from 'path';
import _ from 'lodash';

// Fake ChapterDocumentStore to collect chapters added.
class FakeChapterDocumentStore extends ChapterDocumentStore {
    public chapters: ChapterDocument[] = [];
    async addChapter(chapterDoc: ChapterDocument): Promise<void> {
        this.chapters.push(chapterDoc);
    }
}

const TEST_DB_PATH = path.join(import.meta.dir, 'test-novels.db');

describe('NovelDocumentStore', () => {
    let fakeChapterStore: FakeChapterDocumentStore;
    let novelStore: NovelDocumentStore;

    beforeEach(async () => {
        try {
            await fs.unlink(TEST_DB_PATH);
        } catch{ /* ignore error */ }
        fakeChapterStore = new FakeChapterDocumentStore();
        // Construct novelStore with a given file path and the fake chapter store.
        novelStore = new NovelDocumentStore(TEST_DB_PATH, fakeChapterStore);
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
        const coll = novelStore.db.getCollection('novels');
        expect(coll.findOne({ 'metadata.novelID': computedID })).toBeDefined();
        // Verify that chapters were added and numbered correctly.
        expect(fakeChapterStore.chapters.length).toBeGreaterThan(0);
        _.forEach(fakeChapterStore.chapters, (chapter, index) => {
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
        const coll = novelStore.db.getCollection('novels');
        expect(coll.findOne({ 'metadata.novelID': providedID })).toBeDefined();
        expect(fakeChapterStore.chapters.length).toBe(1);
        const chapter = fakeChapterStore.chapters[0];
        expect(chapter.metadata.chapter).toBe(1);
        expect(chapter.metadata.novelID).toBe(providedID);
    });
});

describe('computeNovelID', () => {
    it('computes a consistent novelID', () => {
        expect(computeNovelID(' John Doe ', ' Test Novel ')).toBe('john_doe_test_novel');
    });
});
