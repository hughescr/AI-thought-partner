import { ChapterSummaryDocument, ChapterSummaryDocumentStore } from '../lib/ChapterSummaryDocumentStore';
import { describe, it, expect, beforeEach, afterEach } from 'bun:test';
import { unlink, access } from 'node:fs/promises';
import { join as pathJoin } from 'node:path';
import { tmpdir } from 'node:os';
import Loki from 'lokijs';

const TEST_DB_PATH = pathJoin(tmpdir(), `test-summaries.db`);

describe('ChapterSummaryDocument', () => {
    it('creates document with chapter metadata', () => {
        const doc = new ChapterSummaryDocument({
            pageContent: 'Test content',
            metadata: { chapter: 1 }
        });

        expect(doc.pageContent).toBe('Test content');
        expect(doc.metadata.chapter).toBe(1);
    });
});

describe('ChapterSummaryDocumentStore', () => {
    let db: Loki;
    let store: ChapterSummaryDocumentStore;

    beforeEach(async () => {
        try {
            await access(TEST_DB_PATH);
            throw new Error(`Test database file ${TEST_DB_PATH} already exists. Aborting.`);
        } catch{ /* file does not exist; continue */ }
        db = new Loki(TEST_DB_PATH, {
            adapter: new Loki.LokiFsAdapter(),
            autosave: true,
            autosaveInterval: 5000,
            autoload: true,
            autoloadCallback: () => ({}),
        });
        store = new ChapterSummaryDocumentStore(db);
    });

    afterEach(async () => {
        try {
            await unlink(TEST_DB_PATH);
        } catch{ /* ignore */ }
    });

    it('stores and retrieves chapter summaries', async () => {
        const doc = new ChapterSummaryDocument({
            pageContent: 'Chapter 1 summary',
            metadata: { chapter: 1 }
        });

        await store.addChapterSummary(doc);
        const retrieved = await store.getChapterSummary(1);

        expect(retrieved?.pageContent).toBe('Chapter 1 summary');
        expect(retrieved?.metadata.chapter).toBe(1);
        await store.close();
    });

    it('returns undefined for non-existent chapters', async () => {
        const result = await store.getChapterSummary(999);
        expect(result).toBeUndefined();
        await store.close();
    });

    it('overwrites existing chapter entries', async () => {
        const doc = new ChapterSummaryDocument({
            pageContent: 'Old summary',
            metadata: { chapter: 2 }
        });

        await store.addChapterSummary(doc);
        doc.pageContent = 'New summary';

        const result = await store.getChapterSummary(2);
        expect(result?.pageContent).toBe('New summary');
        await store.close();
    });

    it('persists data between instances', async () => {
        const firstStore = new ChapterSummaryDocumentStore(db);
        const doc = new ChapterSummaryDocument({
            pageContent: 'Lasting content',
            metadata: { chapter: 3 }
        });

        await firstStore.addChapterSummary(doc);
        await firstStore.close();

        // Create new store instance to verify persistence
        const secondStore = new ChapterSummaryDocumentStore(db);
        const persistedDoc = await secondStore.getChapterSummary(3);

        expect(persistedDoc?.pageContent).toBe('Lasting content');
        await secondStore.close();
    });

    it('handles invalid chapter numbers', async () => {
        expect(await store.getChapterSummary(0)).toBeUndefined();
        expect(await store.getChapterSummary(-1)).toBeUndefined();
        expect(await store.getChapterSummary(NaN)).toBeUndefined();
        await store.close();
    });

    it('throws error when adding document without chapter metadata', async () => {
        // @ts-expect-error: Testing invalid input
        await expect(store.addChapterSummary(new ChapterSummaryDocument({
            pageContent: 'Invalid doc'
        }))).rejects.toThrow();
        await store.close();
    });

    it('auto-saves changes when modifying pageContent on retrieved document', async () => {
        const doc = new ChapterSummaryDocument({
            pageContent: 'Initial content',
            metadata: { chapter: 4 }
        });
        await store.addChapterSummary(doc);

        // Retrieve the document and confirm initial content
        let retrieved = await store.getChapterSummary(4);
        expect(retrieved?.pageContent).toBe('Initial content');

        // Modify the pageContent property on the retrieved document
        if(retrieved) {
            retrieved.pageContent = 'Updated content';
        }

        // Retrieve again to verify the change has been auto-saved
        retrieved = await store.getChapterSummary(4);
        expect(retrieved?.pageContent).toBe('Updated content');
        await store.close();
    });
});
