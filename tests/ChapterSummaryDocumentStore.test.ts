import { ChapterSummaryDocument, ChapterSummaryDocumentStore } from '../lib/ChapterSummaryDocumentStore';
import { ChapterDocument } from '../lib/ChapterDocumentStore';
import { ChapterSummaryGenerator } from '../lib/ChapterSummaryGenerator';
import { describe, it, expect, beforeEach, afterEach } from 'bun:test';
import { access } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join as pathJoin } from 'node:path';
import PouchDB from 'pouchdb';
import find from 'pouchdb-find';
PouchDB.plugin(find);
import { RunnableLambda } from '@langchain/core/runnables';
import _ from 'lodash';

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

const summaryGenerator = new ChapterSummaryGenerator({
    targetSummarySize: 100,
    llm: RunnableLambda.from(_.constant('Concise generated summary')),
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
        store = new ChapterSummaryDocumentStore({ db, summaryGenerator });
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
            pageContent: '# Chapter 1\nChapter 1 content',
            metadata: { novelID: 'test', chapter: 1 }
        });

        await store.addChapterSummary(chapter);
        const retrieved = await store.getChapterSummary(chapter);

        expect(retrieved?.pageContent).toBe('# Chapter 1\nConcise generated summary');
        expect(retrieved?.metadata.chapter).toBe(1);
    });

    it('returns undefined for non-existent chapters', async () => {
        const doc = new ChapterDocument({
            pageContent: '#Chapter 999\nNon-existent chapter',
            metadata: { novelID: 'test', chapter: 999 }
        });
        const result = await store.getChapterSummary(doc);
        expect(result).toBeUndefined();
    });

    it('persists data between instances', async () => {
        const chapter = new ChapterDocument({
            pageContent: '#Chapter 3\nChapter 3 content',
            metadata: { novelID: 'test', chapter: 3 }
        });
        const firstStore = new ChapterSummaryDocumentStore({ db, summaryGenerator });

        await firstStore.addChapterSummary(chapter);

        // Create new store instance to verify persistence
        const secondStore = new ChapterSummaryDocumentStore({ db, summaryGenerator });
        const persistedDoc = await secondStore.getChapterSummary(chapter);

        expect(persistedDoc?.pageContent).toBe('#Chapter 3\nConcise generated summary');
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
