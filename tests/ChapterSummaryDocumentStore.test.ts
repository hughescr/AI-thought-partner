import { ChapterSummaryDocument, ChapterSummaryDocumentStore } from '../lib/ChapterSummaryDocumentStore';
import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import { removeSync } from 'fs-extra';

const TEST_DB_PATH = './test-summaries.db';

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
  beforeEach(() => {
    removeSync(TEST_DB_PATH); // Clean up before each test
  });

  afterEach(() => {
    removeSync(TEST_DB_PATH); // Clean up after each test
  });

  it('stores and retrieves chapter summaries', () => {
    const store = new ChapterSummaryDocumentStore(TEST_DB_PATH);
    const doc = new ChapterSummaryDocument({
      pageContent: 'Chapter 1 summary',
      metadata: { chapter: 1 }
    });

    store.addChapterSummary(doc);
    const retrieved = store.getChapterSummary(1);
    
    expect(retrieved?.pageContent).toBe('Chapter 1 summary');
    expect(retrieved?.metadata.chapter).toBe(1);
  });

  it('returns undefined for non-existent chapters', () => {
    const store = new ChapterSummaryDocumentStore(TEST_DB_PATH);
    expect(store.getChapterSummary(999)).toBeUndefined();
  });

  it('overwrites existing chapter entries', () => {
    const store = new ChapterSummaryDocumentStore(TEST_DB_PATH);
    const initialDoc = new ChapterSummaryDocument({
      pageContent: 'Old summary',
      metadata: { chapter: 2 }
    });
    
    const updatedDoc = new ChapterSummaryDocument({
      pageContent: 'New summary',
      metadata: { chapter: 2 }
    });

    store.addChapterSummary(initialDoc);
    store.addChapterSummary(updatedDoc);
    
    const result = store.getChapterSummary(2);
    expect(result?.pageContent).toBe('New summary');
  });

  it('persists data between instances', async () => {
    const firstStore = new ChapterSummaryDocumentStore(TEST_DB_PATH);
    const doc = new ChapterSummaryDocument({
      pageContent: 'Lasting content',
      metadata: { chapter: 3 }
    });
    
    firstStore.addChapterSummary(doc);
    
    // Create new store instance to verify persistence
    const secondStore = new ChapterSummaryDocumentStore(TEST_DB_PATH);
    const persistedDoc = secondStore.getChapterSummary(3);
    
    expect(persistedDoc?.pageContent).toBe('Lasting content');
  });

  it('handles invalid chapter numbers', () => {
    const store = new ChapterSummaryDocumentStore(TEST_DB_PATH);
    expect(store.getChapterSummary(0)).toBeUndefined();
    expect(store.getChapterSummary(-1)).toBeUndefined();
    expect(store.getChapterSummary(NaN)).toBeUndefined();
  });

  it('throws error when adding document without chapter metadata', () => {
    const store = new ChapterSummaryDocumentStore(TEST_DB_PATH);
    expect(() => {
      // @ts-expect-error: Testing invalid input
      store.addChapterSummary(new ChapterSummaryDocument({
        pageContent: 'Invalid doc'
      }));
    }).toThrow();
  });
});
