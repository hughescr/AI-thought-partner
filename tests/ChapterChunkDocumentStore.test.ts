import { describe, it, expect, beforeEach, afterEach } from 'bun:test';
import Loki from 'lokijs';
import { ChapterChunkDocumentStore } from '../lib/ChapterChunkDocumentStore';
import { unlink } from 'node:fs/promises';

const TEST_DB_PATH = './test-chunks.db';

describe('ChapterChunkDocumentStore', () => {
    let db: Loki;
    let store: ChapterChunkDocumentStore;

    beforeEach(async () => {
        try { await unlink(TEST_DB_PATH) } catch {}
        db = new Loki(TEST_DB_PATH, {
            adapter: new Loki.LokiFsAdapter(),
            autosave: true,
            autosaveInterval: 5000
        });
        store = new ChapterChunkDocumentStore(db);
    });

    afterEach(async () => {
        try { await unlink(TEST_DB_PATH) } catch {}
        await store.close();
    });

    it('stores and retrieves chapter chunks', async () => {
        const content = '# Chapter 1\n\n' + 'text '.repeat(1000);
        await store.addChunksForChapter(1, content);
        const chunks = await store.getChapterChunks(1);
        
        expect(chunks.length).toBeGreaterThan(1);
        expect(chunks[0].metadata.sequence).toBe(1);
        expect(chunks[0].pageContent).toContain('Chapter 1');
    });

    it('deletes all chunks for a chapter', async () => {
        await store.addChunksForChapter(2, 'Short content');
        await store.deleteChapterChapters(2);
        const chunks = await store.getChapterChunks(2);
        expect(chunks).toHaveLength(0);
    });

    it('maintains unique chapter+sequence combinations', async () => {
        await store.addChunksForChapter(3, 'Content');
        expect(() => {
            store.addChunksForChapter(3, 'New content');
        }).toThrow('Duplicate key for property metadata.chapter: 3');
    });
});
