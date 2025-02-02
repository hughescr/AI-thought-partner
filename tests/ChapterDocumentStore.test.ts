import { ChapterDocument, ChapterDocumentStore } from '../lib/ChapterDocumentStore';
import { describe, it, expect, beforeEach, afterEach } from 'bun:test';
import { unlink } from 'node:fs/promises';

const TEST_DB_PATH = './test-chapters.db';

describe('ChapterDocument', () => {
    it('creates document with chapter content and metadata', () => {
        const doc = new ChapterDocument({
            pageContent: '# Chapter 1\n\nIt was a dark and stormy night...',
            metadata: { chapter: 1 }
        });

        expect(doc.pageContent).toContain('stormy night');
        expect(doc.metadata.chapter).toBe(1);
    });
});

describe('ChapterDocumentStore', () => {
    beforeEach(async () => {
        try {
            await unlink(TEST_DB_PATH);
        } catch (e) {}
    });

    afterEach(async () => {
        try {
            await unlink(TEST_DB_PATH);
        } catch (e) {}
    });

    it('stores and retrieves chapters', async () => {
        const store = new ChapterDocumentStore(TEST_DB_PATH);
        const doc = new ChapterDocument({
            pageContent: '# Prologue\n\nOnce upon a time...',
            metadata: { chapter: 0 }
        });

        await store.addChapter(doc);
        const retrieved = await store.getChapter(0);

        expect(retrieved?.pageContent).toContain('Once upon');
        expect(retrieved?.metadata.chapter).toBe(0);
        await store.close();
    });

    it('overwrites existing chapters', async () => {
        const store = new ChapterDocumentStore(TEST_DB_PATH);
        const doc = new ChapterDocument({
            pageContent: 'Original content',
            metadata: { chapter: 5 }
        });

        await store.addChapter(doc);
        doc.pageContent = 'Revised content';

        const updated = await store.getChapter(5);
        expect(updated?.pageContent).toBe('Revised content');
        await store.close();
    });

    it('persists chapters across instances', async () => {
        const firstStore = new ChapterDocumentStore(TEST_DB_PATH);
        const doc = new ChapterDocument({
            pageContent: '# Epilogue\n\nAnd they lived...',
            metadata: { chapter: 99 }
        });

        await firstStore.addChapter(doc);
        await firstStore.close();

        const secondStore = new ChapterDocumentStore(TEST_DB_PATH);
        const persisted = await secondStore.getChapter(99);

        expect(persisted?.pageContent).toContain('lived');
        await secondStore.close();
    });

    it('rejects documents without chapter metadata', async () => {
        const store = new ChapterDocumentStore(TEST_DB_PATH);

        // @ts-expect-error: Testing invalid input
        await expect(store.addChapter(new ChapterDocument({
            pageContent: 'Invalid content'
        }))).rejects.toThrow('chapter metadata is required');

        await store.close();
    });

    it('auto-updates document changes', async () => {
        const store = new ChapterDocumentStore(TEST_DB_PATH);
        const doc = new ChapterDocument({
            pageContent: 'Initial version',
            metadata: { chapter: 10 }
        });

        await store.addChapter(doc);
        doc.pageContent = 'Updated version';

        const result = await store.getChapter(10);
        expect(result?.pageContent).toBe('Updated version');
        await store.close();
    });
});
