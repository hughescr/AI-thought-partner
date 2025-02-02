import { describe, it, expect, beforeEach, afterEach } from 'bun:test';
import { unlink } from 'node:fs/promises';
import path from 'node:path';
import { Document } from '@langchain/core/documents';
import { ChapterDocument, ChapterDocumentStore } from '../lib/ChapterDocumentStore';
import { ChapterSummaryGenerator } from '../lib/ChapterSummaryGenerator';
import { RunnableLambda } from '@langchain/core/runnables';

class MockSummaryGenerator extends ChapterSummaryGenerator {
    public async generateSummary(chapter: Document) {
        return new Document({
            pageContent: 'Concise generated summary',
            metadata: { chapter: chapter.metadata.chapter }
        });
    }
}

const TEST_DB_PATH = path.join(import.meta.dir, 'test-chapters.db');

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
    let mockSummaryGenerator: MockSummaryGenerator;
    beforeEach(async () => {
        mockSummaryGenerator = new MockSummaryGenerator({ llm: RunnableLambda.from((input: { text: string }) => input.text), targetSummarySize: 100 });
        try {
            await unlink(TEST_DB_PATH);
        } catch{ /* ignore error */ }
    });

    afterEach(async () => {
        try {
            await unlink(TEST_DB_PATH);
        } catch{ /* ignore error */ }
    });

    it('stores and retrieves chapters', async () => {
        const store = new ChapterDocumentStore(TEST_DB_PATH, mockSummaryGenerator);
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
        const store = new ChapterDocumentStore(TEST_DB_PATH, mockSummaryGenerator);
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
        const firstStore = new ChapterDocumentStore(TEST_DB_PATH, mockSummaryGenerator);
        const doc = new ChapterDocument({
            pageContent: '# Epilogue\n\nAnd they lived...',
            metadata: { chapter: 99 }
        });

        await firstStore.addChapter(doc);
        await firstStore.close();

        const secondStore = new ChapterDocumentStore(TEST_DB_PATH, mockSummaryGenerator);
        const persisted = await secondStore.getChapter(99);

        expect(persisted?.pageContent).toContain('lived');
        await secondStore.close();
    });

    it('rejects documents without chapter metadata', async () => {
        const store = new ChapterDocumentStore(TEST_DB_PATH, mockSummaryGenerator);

        // @ts-expect-error: Testing invalid input
        await expect(store.addChapter(new ChapterDocument({
            pageContent: 'Invalid content'
        }))).rejects.toThrow('chapter metadata is required');

        await store.close();
    });

    it('auto-updates document changes', async () => {
        const store = new ChapterDocumentStore(TEST_DB_PATH, mockSummaryGenerator);
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

    it('returns undefined for a non-existent chapter', async () => {
        const store = new ChapterDocumentStore(TEST_DB_PATH, mockSummaryGenerator);
        const nonExistent = await store.getChapter(12345);
        expect(nonExistent).toBeUndefined();
        await store.close();
    });

    // Replace the "updates summary when re-adding the same chapter" test
    it('throws error when re-adding an already added document', async () => {
        const store = new ChapterDocumentStore(TEST_DB_PATH, mockSummaryGenerator);
        const doc = new ChapterDocument({
            pageContent: 'Initial content',
            metadata: { chapter: 7 }
        });
        await store.addChapter(doc);
        // Attempt to re-add should throw an error.
        await expect(store.addChapter(doc)).rejects.toThrow('Document is already in collection, please use update()');
        await store.close();
    });

    it('generates and retrieves chapter summary', async () => {
        const store = new ChapterDocumentStore(TEST_DB_PATH, mockSummaryGenerator);
        const doc = new ChapterDocument({
            pageContent: 'Content for summary test',
            metadata: { chapter: 15 }
        });
        await store.addChapter(doc);
        const summary = await store.getChapterSummary(15);
        expect(summary).toBeDefined();
        expect(summary?.pageContent).toBe('Concise generated summary');
        await store.close();
    });
});
