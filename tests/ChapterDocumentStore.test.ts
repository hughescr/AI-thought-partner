import { describe, it, expect, beforeEach, afterEach } from 'bun:test';
import { unlink, access } from 'node:fs/promises';
import { join as pathJoin } from 'node:path';
import { tmpdir } from 'node:os';
import { ChapterDocument, ChapterDocumentStore } from '../lib/ChapterDocumentStore';
import { ChapterSummaryGenerator } from '../lib/ChapterSummaryGenerator';
import { RunnableLambda } from '@langchain/core/runnables';
import _ from 'lodash';
import Loki from 'lokijs';

const TEST_DB_PATH = pathJoin(tmpdir(), `test-chapters.db`);

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
    let db: Loki;
    beforeEach(async () => {
        summaryGenerator = new ChapterSummaryGenerator({
            llm: RunnableLambda.from(_.constant('Concise generated summary')),
            targetSummarySize: 100
        });
        try {
            await access(TEST_DB_PATH);
            throw new Error(`Test database file ${TEST_DB_PATH} already exists. Aborting.`);
        } catch{ /* ignore error */ }
        db = new Loki(TEST_DB_PATH, {
            adapter: new Loki.LokiFsAdapter(),
            autoload: true,
            autosave: true,
            autosaveInterval: 5000,
        });
    });

    afterEach(async () => {
        try {
            await unlink(TEST_DB_PATH);
        } catch{ /* ignore error */ }
    });

    it('stores and retrieves chapters', async () => {
        const store = new ChapterDocumentStore(db, summaryGenerator);
        const doc = new ChapterDocument({
            pageContent: '# Prologue\n\nOnce upon a time...',
            metadata: { novelID: 'test', chapter: 0 }
        });

        await store.addChapter(doc);
        const retrieved = await store.getChapter(0);

        expect(retrieved?.pageContent).toContain('Once upon');
        expect(retrieved?.metadata.chapter).toBe(0);
        await store.close();
    });

    it('overwrites existing chapters', async () => {
        const store = new ChapterDocumentStore(db, summaryGenerator);
        const doc = new ChapterDocument({
            pageContent: '# Chapter 5\nOriginal content',
            metadata: { novelID: 'test', chapter: 5 }
        });

        await store.addChapter(doc);
        doc.pageContent = '# Chapter 5\nRevised content';

        const updated = await store.getChapter(5);
        expect(updated?.pageContent).toBe('# Chapter 5\nRevised content');
        await store.close();
    });

    it('persists chapters across instances', async () => {
        const firstStore = new ChapterDocumentStore(db, summaryGenerator);
        const doc = new ChapterDocument({
            pageContent: '# Epilogue\n\nAnd they lived...',
            metadata: { novelID: 'test', chapter: 99 }
        });

        await firstStore.addChapter(doc);
        await firstStore.close();

        const secondStore = new ChapterDocumentStore(db, summaryGenerator);
        const persisted = await secondStore.getChapter(99);

        expect(persisted?.pageContent).toContain('lived');
        await secondStore.close();
    });

    it('rejects documents without chapter metadata', async () => {
        const store = new ChapterDocumentStore(db, summaryGenerator);

        // @ts-expect-error: Testing invalid input
        await expect(store.addChapter(new ChapterDocument({
            pageContent: 'Invalid content'
        }))).rejects.toThrow('chapter metadata is required');

        await store.close();
    });

    it('auto-updates document changes', async () => {
        const store = new ChapterDocumentStore(db, summaryGenerator);
        const doc = new ChapterDocument({
            pageContent: '# Chapter 10\nInitial version',
            metadata: { novelID: 'test', chapter: 10 }
        });

        await store.addChapter(doc);
        doc.pageContent = '# Chapter 10\nUpdated version';

        const result = await store.getChapter(10);
        expect(result?.pageContent).toBe('# Chapter 10\nUpdated version');
        await store.close();
    });

    it('returns undefined for a non-existent chapter', async () => {
        const store = new ChapterDocumentStore(db, summaryGenerator);
        const nonExistent = await store.getChapter(12345);
        expect(nonExistent).toBeUndefined();
        await store.close();
    });

    // Replace the "updates summary when re-adding the same chapter" test
    it('throws error when re-adding an already added document', async () => {
        const store = new ChapterDocumentStore(db, summaryGenerator);
        const doc = new ChapterDocument({
            pageContent: '# Chapter 7\nInitial content',
            metadata: { novelID: 'test', chapter: 7 }
        });
        await store.addChapter(doc);
        // Attempt to re-add should throw an error.
        await expect(store.addChapter(doc)).rejects.toThrow('Document is already in collection, please use update()');
        await store.close();
    });

    it('generates and retrieves chapter summary', async () => {
        const store = new ChapterDocumentStore(db, summaryGenerator);
        const doc = new ChapterDocument({
            pageContent: '# Chapter 15\nContent for summary test',
            metadata: { novelID: 'test', chapter: 15 }
        });
        await store.addChapter(doc);
        const summary = await store.getChapterSummary(15);
        expect(summary).toBeDefined();
        expect(summary?.pageContent).toBe('# Chapter 15\nConcise generated summary');
        await store.close();
    });

    // For the dynamic summary test, instantiate a local generator that reflects the chapter content.
    it('updates chapter summary when chapter content changes', async () => {
        let text = 'Initial chapter content';
        const dynamicGenerator = new ChapterSummaryGenerator({
            llm: RunnableLambda.from(() => ({ content: [{ text: `Summary: ${text}`, type: 'text' }] })),
            targetSummarySize: 100
        });
        const store = new ChapterDocumentStore(db, dynamicGenerator);
        const doc = new ChapterDocument({
            pageContent: `# Chapter 21\n${text}`,
            metadata: { novelID: 'test', chapter: 21 }
        });
        await store.addChapter(doc);

        const summary1 = await store.getChapterSummary(21);
        expect(summary1?.pageContent).toBe('# Chapter 21\nSummary: Initial chapter content');

        // Update chapter content
        text = 'Updated chapter content';
        doc.pageContent = `# Chapter 21\n${text}`;
        // Wait for async update
        await new Promise(resolve => setTimeout(resolve, 150));
        const summary2 = await store.getChapterSummary(21);
        expect(summary2?.pageContent).toBe('# Chapter 21\nSummary: Updated chapter content');
        await store.close();
    });

    it('attaches auto-update hook to document after adding chapter', async () => {
        const store = new ChapterDocumentStore(db, summaryGenerator);
        const doc = new ChapterDocument({
            pageContent: '# Test Chapter\nTest content',
            metadata: { novelID: 'test', chapter: 42 }
        });
        await store.addChapter(doc);

        // Verify that the doc now has an own property descriptor for pageContent with a setter
        const descriptor = Object.getOwnPropertyDescriptor(doc, 'pageContent');
        expect(descriptor).toBeDefined();
        expect(typeof descriptor.set).toBe('function');
    });
});
