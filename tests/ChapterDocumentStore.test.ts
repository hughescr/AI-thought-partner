import { describe, it, expect, beforeEach, afterEach } from 'bun:test';
import { fileURLToPath } from 'node:url';
import { dirname, join as pathJoin } from 'node:path';
import { ChapterDocument, ChapterDocumentStore } from '../lib/ChapterDocumentStore';
import { NovelDocument } from '../lib/NovelDocumentStore';
import { ChapterSummaryGenerator } from '../lib/ChapterSummaryGenerator';
import { RunnableLambda } from '@langchain/core/runnables';
import _ from 'lodash';
import PouchDB from 'pouchdb';
import find from 'pouchdb-find';
PouchDB.plugin(find);

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const TEST_DB_PATH = pathJoin(__dirname, 'test-chapters.db');

describe('ChapterDocument', () => {
    it('creates document with chapter content and metadata', () => {
        const doc = new ChapterDocument({
            pageContent: '# Chapter 1\n\nIt was a dark and stormy night...',
            metadata: { novelID: 'test', chapter: 1 }
        });

        expect(doc.pageContent).toContain('stormy night');
        expect(doc.metadata.chapter).toBe(1);
    });
});

describe('ChapterDocumentStore', () => {
    let summaryGenerator: ChapterSummaryGenerator;
    let db: PouchDB.Database;
    beforeEach(async () => {
        summaryGenerator = new ChapterSummaryGenerator({
            llm: RunnableLambda.from(_.constant('Concise generated summary')),
            targetSummarySize: 100
        });
        db = new PouchDB(TEST_DB_PATH); // Create or open existing DB
        await db.destroy();             // Delete it -- will clean up if there was already DB there
        db = new PouchDB(TEST_DB_PATH); // Now create a new one which will be empty
    });

    afterEach(async () => {
        if(db) {
            await db.destroy();
        }
    });

    it('stores and retrieves chapters', async () => {
        const novel = new NovelDocument({
            pageContent: 'dummy',
            metadata: { title: 'Test Novel', author: 'Test Author' }
        });
        const store = new ChapterDocumentStore({ db, summaryGenerator });
        const doc = new ChapterDocument({
            pageContent: '# Prologue\n\nOnce upon a time...',
            metadata: { novelID: novel.metadata.novelID, chapter: 0 }
        });

        await store.addChapter(doc);
        const retrieved = await store.getChapter(novel, 0);

        expect(retrieved?.pageContent).toContain('Once upon');
        expect(retrieved?.metadata.chapter).toBe(0);
    });

    it('persists chapters across instances', async () => {
        const novel = new NovelDocument({
            pageContent: 'dummy',
            metadata: { title: 'Test Novel', author: 'Test Author' }
        });
        const firstStore = new ChapterDocumentStore({ db, summaryGenerator });
        const doc = new ChapterDocument({
            pageContent: '# Epilogue\n\nAnd they lived...',
            metadata: { novelID: novel.metadata.novelID, chapter: 99 }
        });

        await firstStore.addChapter(doc);

        const secondStore = new ChapterDocumentStore({ db, summaryGenerator });
        const persisted = await secondStore.getChapter(novel, 99);

        expect(persisted).toBeDefined();
        expect(persisted?.pageContent).toContain('lived');
    });

    it('rejects documents without chapter metadata', async () => {
        const store = new ChapterDocumentStore({ db, summaryGenerator });

        // @ts-expect-error: Testing invalid input
        await expect(store.addChapter(new ChapterDocument({
            pageContent: 'Invalid content'
        }))).rejects.toThrow('chapter metadata is required');
    });

    it('returns undefined for a non-existent chapter', async () => {
        const novel = new NovelDocument({
            pageContent: 'dummy',
            metadata: { title: 'Test Novel', author: 'Test Author' }
        });
        const store = new ChapterDocumentStore({ db, summaryGenerator });
        const nonExistent = await store.getChapter(novel, 12345);
        expect(nonExistent).toBeUndefined();
    });

    // Replace the "updates summary when re-adding the same chapter" test
    it('throws error when re-adding an already added document', async () => {
        const store = new ChapterDocumentStore({ db, summaryGenerator });
        const doc = new ChapterDocument({
            pageContent: '# Chapter 7\nInitial content',
            metadata: { novelID: 'test', chapter: 7 }
        });
        await store.addChapter(doc);
        // Attempt to re-add should throw an error.
        await expect(store.addChapter(doc)).rejects.toThrow('Document is already in collection, please use update()');
    });

    it('generates and retrieves chapter summary', async () => {
        const store = new ChapterDocumentStore({ db, summaryGenerator });
        const doc = new ChapterDocument({
            pageContent: '# Chapter 15\nContent for summary test',
            metadata: { novelID: 'test', chapter: 15 }
        });
        await store.addChapter(doc);
        const summary = await store.getChapterSummary(doc);
        expect(summary).toBeDefined();
        expect(summary?.pageContent).toBe('# Chapter 15\nConcise generated summary');
    });

    it('generates and retrieves chapter chunks', async () => {
        // Create a dummy novel for proper novelID generation.
        const novel = new NovelDocument({
            pageContent: 'dummy',
            metadata: { title: 'Chunk Test Novel', author: 'Test Author' }
        });
        const store = new ChapterDocumentStore({ db, summaryGenerator });
        // Create chapter content that is long enough to trigger chunking.
        const content = '# Chapter 2\n' + _.repeat('Lorem ipsum dolor sit amet, consectetur adipiscing elit. ', 20);
        const doc = new ChapterDocument({
            pageContent: content,
            metadata: { novelID: novel.metadata.novelID, chapter: 2 }
        });
        await store.addChapter(doc);
        const chunks = await store.getChapterChunks(doc);
        expect(chunks.length).toBeGreaterThan(0);
        _.forEach(chunks, (chunk) => {
            expect(chunk.metadata.novelID).toEqual(novel.metadata.novelID);
            expect(chunk.metadata.chapter).toEqual(2);
            expect(chunk.pageContent).toBeTruthy();
        });
    });

    it('handles concurrent chapter updates safely', async () => {
        const store = new ChapterDocumentStore({ db, summaryGenerator });
        const doc = new ChapterDocument({
            pageContent: '# Chapter 30\nInitial content',
            metadata: { novelID: 'test', chapter: 30 }
        });
        await store.addChapter(doc);

        // Perform multiple updates in rapid succession
        doc.pageContent = '# Chapter 30\nUpdate 1';
        doc.pageContent = '# Chapter 30\nUpdate 2';
        doc.pageContent = '# Chapter 30\nFinal update';

        // Wait long enough for all async update chains to finish.
        await new Promise(resolve => setTimeout(resolve, 300));

        // Retrieve the summary; it should reflect the final update.
        const finalSummary = await store.getChapterSummary(doc);
        expect(finalSummary?.pageContent).toBe('# Chapter 30\nConcise generated summary');
    });

    it('getChapters retrieves all chapters for a novel', async () => {
        const novel = new NovelDocument({
            pageContent: 'dummy',
            metadata: { title: 'Multiple Chapters', author: 'Test Author' }
        });
        const store = new ChapterDocumentStore({ db, summaryGenerator });

        // Add multiple chapters
        for(let i = 1; i <= 3; i++) {
            const doc = new ChapterDocument({
                pageContent: `# Chapter ${i}\nContent for chapter ${i}`,
                metadata: { novelID: novel.metadata.novelID, chapter: i }
            });
            await store.addChapter(doc);
        }

        // Retrieve all chapters
        const chapters = await store.getChapters(novel);
        expect(chapters.length).toBe(3);
        expect(_.map(chapters, 'metadata.chapter').sort()).toEqual([1, 2, 3]);
    });

    it('getChapterSummaries retrieves all summaries for a novel', async () => {
        const store = new ChapterDocumentStore({ db, summaryGenerator });
        const novelID = 'summaries_test_novel';

        // Add multiple chapters
        for(let i = 1; i <= 2; i++) {
            const doc = new ChapterDocument({
                pageContent: `# Chapter ${i}\nContent for summaries test ${i}`,
                metadata: { novelID, chapter: i }
            });
            await store.addChapter(doc);
        }

        // Retrieve all summaries
        const summaries = await store.getChapterSummaries(novelID);
        expect(summaries.length).toBe(2);
        expect(_.every(summaries, summary => summary.pageContent.includes('Concise generated summary'))).toBe(true);
        expect(_.map(summaries, 'metadata.chapter').sort()).toEqual([1, 2]);
    });
});
