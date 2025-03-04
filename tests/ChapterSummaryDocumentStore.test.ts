import { ChapterSummaryDocument, ChapterSummaryDocumentStore } from '../lib/ChapterSummaryDocumentStore';
import { ChapterDocument } from '../lib/ChapterDocumentStore';
import { ChapterSummaryGenerator } from '../lib/ChapterSummaryGenerator';
import { describe, it, expect, beforeEach, afterEach } from 'bun:test';
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
        db = new PouchDB(TEST_DB_PATH); // Create or open existing DB
        await db.destroy();             // Delete it -- will clean up if there was already DB there
        db = new PouchDB(TEST_DB_PATH); // Now create a new one which will be empty
        store = new ChapterSummaryDocumentStore({ db, summaryGenerator });
    });

    afterEach(async () => {
        if(db) {
            await db.destroy();
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

    it('getChapterSummaries returns all summaries for a novel', async () => {
        const novelID = 'novel_with_multiple_chapters';
        
        // Add summaries for multiple chapters
        for (let i = 1; i <= 3; i++) {
            const chapter = new ChapterDocument({
                pageContent: `# Chapter ${i}\nContent for chapter ${i}`,
                metadata: { novelID, chapter: i }
            });
            await store.addChapterSummary(chapter);
        }
        
        // Retrieve all summaries
        const summaries = await store.getChapterSummaries(novelID);
        expect(summaries.length).toBe(3);
        expect(_.map(summaries, 'metadata.chapter').sort()).toEqual([1, 2, 3]);
        expect(_.every(summaries, summary => summary.pageContent.includes('Concise generated summary'))).toBe(true);
    });

    it('replaces existing summary when adding a summary for the same chapter', async () => {
        const chapter = new ChapterDocument({
            pageContent: '# Chapter 5\nInitial content',
            metadata: { novelID: 'replacement_test', chapter: 5 }
        });
        
        // Add initial summary
        await store.addChapterSummary(chapter);
        
        // Change chapter content and regenerate summary
        chapter.pageContent = '# Chapter 5\nUpdated content';
        await store.addChapterSummary(chapter);
        
        // Get the summary - there should only be one
        const summary = await store.getChapterSummary(chapter);
        expect(summary).toBeDefined();
        
        // Check in database to confirm only one exists
        const response = await db.find({
            selector: {
                'metadata.docType': 'summary',
                'metadata.novelID': 'replacement_test',
                'metadata.chapter': 5
            }
        });
        expect(response.docs.length).toBe(1);
    });
});
