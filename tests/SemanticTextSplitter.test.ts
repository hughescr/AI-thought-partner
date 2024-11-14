import { test, expect, describe, beforeEach } from 'bun:test';
import { SemanticTextSplitter } from '../lib/SemanticTextSplitter';
import { Embeddings } from '@langchain/core/embeddings';
import _ from 'lodash';

class ConcreteEmbeddings extends Embeddings {
    constructor(params: EmbeddingsParams) {
        super();
    }

    embedDocuments(texts: string[]): Promise<number[][]> {
        return Promise.resolve(texts.map(text => _.fill(Array(512), 0.5)));
    }

    embedQuery(text: string): Promise<number[]> {
        return Promise.resolve(_.fill(Array(512), 0.5));
    }
}

describe('SemanticTextSplitter', () => {
    let embeddings: Embeddings;
    let splitter: SemanticTextSplitter;

    beforeEach(() => {
        embeddings = new ConcreteEmbeddings();
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
