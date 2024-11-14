import { test, expect, mock, describe, beforeEach, fn } from 'bun:test';
import { SemanticTextSplitter } from '../lib/SemanticTextSplitter';
import { Embeddings } from '@langchain/core/embeddings';
import _ from 'lodash';

// Mock the Embeddings class
mock('@langchain/core/embeddings', () => {
    return {
        Embeddings: fn().mockImplementation(() => {
            return {
                embedDocuments: fn((texts: string[]) => {
                    return Promise.resolve(texts.map(text => _.fill(Array(512), 0.5)));
                })
            };
        })
    };
    });

    test('should create merged chunks', () => {
        const chunks = ['chunk1', 'chunk2', 'chunk3'];
        const mergedChunks = splitter['createMergedChunks'](chunks);
        expect(mergedChunks).toEqual(['chunk1chunk2', 'chunk2chunk3']);
    });

    test('should embed merged chunks', async () => {
        const mergedChunks = ['chunk1chunk2', 'chunk2chunk3'];
        const embeddings = await splitter['embedMergedChunks'](mergedChunks);
        expect(embeddings.length).toBe(2);
    });

    test('should create final chunks', async () => {
        const initialChunks = ['chunk1', 'chunk2', 'chunk3'];
        const finalChunks = await splitter['createFinalChunks'](initialChunks);
        expect(finalChunks.length).toBeGreaterThan(0);
    });
});

describe('SemanticTextSplitter', () => {
    let embeddings: Embeddings;
    let splitter: SemanticTextSplitter;

    beforeEach(() => {
        embeddings = new Embeddings();
        splitter = new SemanticTextSplitter({
            embeddings,
            chunkSize: 512,
            embeddingBatchSize: 16,
            initialChunkSize: 32,
            showProgress: false
        });
    });

    test('should split text into chunks', async () => {
        const text = "This is a test. This is only a test.";
        const chunks = await splitter.splitText(text);
        expect(chunks.length).toBeGreaterThan(0);
    });

    test('should handle empty text', async () => {
        const text = "";
        const chunks = await splitter.splitText(text);
        expect(chunks).toEqual([]);
    });

    test('should handle text smaller than initial chunk size', async () => {
        const text = "Short text.";
        const chunks = await splitter.splitText(text);
        expect(chunks).toEqual([text]);
    });

    test('should merge chunks based on similarity', async () => {
        const text = "This is a test. This is only a test. This is another test.";
        const chunks = await splitter.splitText(text);
        expect(chunks.length).toBeLessThan(3);
    });
});
