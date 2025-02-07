import { ChapterSummaryDocument, ChapterSummaryDocumentStore } from '../lib/ChapterSummaryDocumentStore';
import { ChapterDocument } from '../lib/ChapterDocumentStore';
import { describe, it, expect, beforeEach, afterEach } from 'bun:test';
import { access } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join as pathJoin } from 'node:path';
import PouchDB from 'pouchdb';
import find from 'pouchdb-find';
PouchDB.plugin(find);

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const TEST_DB_PATH = pathJoin(__dirname, 'test-summaries.db');

describe('ChapterSummaryDocument', () => {
    it('creates document with chapter metadata', () => {
        const doc = new ChapterSummaryDocument({
            pageContent: 'Test content',
            metadata: { novelID: 'test', chapter: 1 }
        });

        expect(doc.pageContent).toBe('Test content');
        expect(doc.metadata.chapter).toBe(1);
    });
});

describe('ChapterSummaryDocumentStore', () => {
    let db: PouchDB.Database;
    let store: ChapterSummaryDocumentStore;

    beforeEach(async () => {
        try {
            await access(TEST_DB_PATH);
            throw new Error(`Test database file ${TEST_DB_PATH} already exists. Aborting.`);
        } catch{ /* file does not exist; continue */ }
        db = new PouchDB(TEST_DB_PATH);
        store = new ChapterSummaryDocumentStore({ db });
    });

    afterEach(async () => {
        if(db) {
            try {
                await db.destroy();
            } catch{
                // Ignore errors
            }
        }
    });

    it('stores and retrieves chapter summaries', async () => {
        const chapter = new ChapterDocument({
            pageContent: 'Chapter 1 content',
            metadata: { novelID: 'test', chapter: 1 }
        });
        const doc = new ChapterSummaryDocument({
            pageContent: 'Chapter 1 summary',
            metadata: { novelID: 'test', chapter: 1 }
        });

        await store.addChapterSummary(doc);
        const retrieved = await store.getChapterSummary(chapter);

        expect(retrieved?.pageContent).toBe('Chapter 1 summary');
        expect(retrieved?.metadata.chapter).toBe(1);
    });

    it('returns undefined for non-existent chapters', async () => {
        const doc = new ChapterDocument({
            pageContent: 'Non-existent chapter',
            metadata: { novelID: 'test', chapter: 999 }
        });
        const result = await store.getChapterSummary(doc);
        expect(result).toBeUndefined();
    });

    it('persists data between instances', async () => {
        const chapter = new ChapterDocument({
            pageContent: 'Chapter 3 content',
            metadata: { novelID: 'test', chapter: 3 }
        });
        const firstStore = new ChapterSummaryDocumentStore({ db });
        const doc = new ChapterSummaryDocument({
            pageContent: 'Lasting content',
            metadata: { novelID: 'test', chapter: 3 }
        });

        await firstStore.addChapterSummary(doc);

        // Sleep for 200ms to allow autosave to complete
        await new Promise(resolve => setTimeout(resolve, 200));

        // Create new store instance to verify persistence
        const secondStore = new ChapterSummaryDocumentStore({ db });
        const persistedDoc = await secondStore.getChapterSummary(chapter);

        expect(persistedDoc?.pageContent).toBe('Lasting content');
    });

    it('handles invalid chapter numbers', async () => {
        const chapter = new ChapterDocument({
            pageContent: 'Invalid chapter',
            metadata: { novelID: 'test', chapter: 0 }
        });
        expect(await store.getChapterSummary(chapter)).toBeUndefined();
        chapter.metadata.chapter = -1;
        expect(await store.getChapterSummary(chapter)).toBeUndefined();
        chapter.metadata.chapter = NaN;
        expect(await store.getChapterSummary(chapter)).toBeUndefined();
    });

    it('throws error when adding document without chapter metadata', async () => {
        // @ts-expect-error: Testing invalid input
        await expect(store.addChapterSummary(new ChapterSummaryDocument({
            pageContent: 'Invalid doc'
        }))).rejects.toThrow();
    });
});
