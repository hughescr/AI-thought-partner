import { describe, it, expect, beforeEach, afterEach } from 'bun:test';
import PouchDB from 'pouchdb';
import find from 'pouchdb-find';
PouchDB.plugin(find);
import { ChapterChunkDocumentStore } from '../lib/ChapterChunkDocumentStore';
import { access } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join as pathJoin } from 'node:path';
import _ from 'lodash';
import { ChapterDocument } from '../lib/ChapterDocumentStore';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const TEST_DB_PATH = pathJoin(__dirname, 'test-chunks.db');

class DummyTextSplitter {
    async splitText(text: string): Promise<string[]> {
        // split on newline and discard empty lines
        return _(text).split('\n').filter(line => _.trim(line).length > 0).value();
    }
}

describe('ChapterChunkDocumentStore', () => {
    let db: PouchDB.Database;
    let store: ChapterChunkDocumentStore;

    beforeEach(async () => {
        try {
            await access(TEST_DB_PATH);
            throw new Error(`Test database file ${TEST_DB_PATH} already exists. Aborting.`);
        } catch{ /* file does not exist; continue */ }
        db = new PouchDB(TEST_DB_PATH);
        store = new ChapterChunkDocumentStore({ db, textSplitter: new DummyTextSplitter() });
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

    it('stores and retrieves chapter chunks', async () => {
        const content = '# Chapter 1\n\n' + _.repeat('text ', 1000);
        const chapterDoc = new ChapterDocument({ pageContent: content, metadata: { novelID: 'unknown', chapter: 1 } });
        await store.addChunksForChapter(chapterDoc);
        const chunks = await store.getChapterChunks(chapterDoc);

        expect(chunks.length).toBeGreaterThan(1);
        expect(chunks[0].metadata.sequence).toBe(1);
        expect(chunks[0].pageContent).toContain('Chapter 1');
        expect(chunks[0].metadata.docType).toBe('chapter_chunk');
        expect(chunks[0].metadata.novelID).toBeDefined();
    });

    it('deletes all chunks for a chapter', async () => {
        const chapterDoc = new ChapterDocument({ pageContent: 'Short content', metadata: { novelID: 'unknown', chapter: 2 } });
        await store.addChunksForChapter(chapterDoc);
        await store.deleteChapterChunks(chapterDoc);
        const chunks = await store.getChapterChunks(chapterDoc);
        expect(chunks).toHaveLength(0);
    });

    it('maintains unique chapter+sequence combinations', async () => {
        const chapterDoc1 = new ChapterDocument({ pageContent: 'Content', metadata: { novelID: 'unknown', chapter: 3 } });
        await store.addChunksForChapter(chapterDoc1);
        const chapterDoc2 = new ChapterDocument({ pageContent: 'New content', metadata: { novelID: 'unknown', chapter: 3 } });
        await expect(store.addChunksForChapter(chapterDoc2))
            .rejects.toThrow('Duplicate key for properties metadata.chapter, metadata.sequence');
    });

    it('maintains proper sequence numbering', async () => {
        const content = 'Chunk1\nChunk2\nChunk3';
        const chapterDoc = new ChapterDocument({ pageContent: content, metadata: { novelID: 'unknown', chapter: 4 } });
        await store.addChunksForChapter(chapterDoc);
        const chunks = await store.getChapterChunks(chapterDoc);
        expect(_.map(chunks, 'metadata.sequence')).toEqual([1, 2, 3]);
    });

    it('persists novelID in metadata when available', async () => {
        const storeWithID = new ChapterChunkDocumentStore({ db, textSplitter: new DummyTextSplitter() });
        const chapterDoc = new ChapterDocument({ pageContent: 'Content', metadata: { novelID: 'test-novel-123', chapter: 5 } });
        await storeWithID.addChunksForChapter(chapterDoc);
        const chunks = await storeWithID.getChapterChunks(chapterDoc);
        expect(chunks[0].metadata.novelID).toBe('test-novel-123');
    });

    it('handles empty chapter content gracefully', async () => {
        const chapterDoc = new ChapterDocument({ pageContent: '', metadata: { novelID: 'unknown', chapter: 6 } });
        await expect(store.addChunksForChapter(chapterDoc))
            .rejects.toThrow('No text chunks generated');
    });

    it('persists chunks across store instances', async () => {
        const chapterDoc = new ChapterDocument({ pageContent: 'Persisted content', metadata: { novelID: 'unknown', chapter: 8 } });
        await store.addChunksForChapter(chapterDoc);
        await store.close();
        const newStore = new ChapterChunkDocumentStore({ db });
        const chunks = await newStore.getChapterChunks(chapterDoc);
        expect(chunks).toHaveLength(1);
    });

    it('prevents duplicate sequences across batches', async () => {
        const chapterDoc1 = new ChapterDocument({ pageContent: 'Batch 1', metadata: { novelID: 'unknown', chapter: 9 } });
        await store.addChunksForChapter(chapterDoc1);
        const chapterDoc2 = new ChapterDocument({ pageContent: 'Batch 2', metadata: { novelID: 'unknown', chapter: 9 } });
        await expect(store.addChunksForChapter(chapterDoc2))
            .rejects.toThrow('Duplicate key for properties metadata.chapter, metadata.sequence');
    });
});
