import _, { chunk } from 'lodash';
import { SingleBar, Presets } from 'cli-progress';

import { TextSplitter, RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { getEncoding } from "@langchain/core/utils/tiktoken";

/**
 * This class first uses a RecursiveCharacterTextSplitter to split the documents into tiny chunks, then re-merges
 * those chunks based on semantic similarity. It will stop when chunks reach the maximum token length or when
 * the similarity drops below some threshold, leaving a set of chunks which have internal semantic similarity
 */
export class SemanticTextSplitter extends TextSplitter {
    static lc_name() {
        return "SemanticTextSplitter";
    }
    constructor(fields) {
        super({ chunkSize: 1000, ...fields, chunkOverlap: 0 });
        Object.defineProperty(this, "showProgress", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: false,
        });
        Object.defineProperty(this, "embeddings", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0,
        });
        Object.defineProperty(this, "tokenizer", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0,
        });
        Object.defineProperty(this, "encodingName", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: 'gpt2',
        });
        Object.defineProperty(this, "allowedSpecial", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: [],
        });
        Object.defineProperty(this, "disallowedSpecial", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: 'all',
        });
        Object.defineProperty(this, "initialChunkSize", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: 50,
        });
        this.showProgress = fields?.showProgress ?? false;
        this.embeddings = fields?.embeddings;
        this.initialChunkSize = fields?.initialChunkSize ?? 50;
        this.encodingName = fields?.encodingName ?? 'gpt2';
        this.allowedSpecial = fields?.allowedSpecial ?? [];
        this.disallowedSpecial = fields?.disallowedSpecial ?? 'all';
    }


    /**
    * Calculate cosine similarity between two vectors --
    */
    cosineSimilarity(vecA, vecB) {
        let dotProduct = 0.0;
        let normA = 0.0;
        let normB = 0.0;

        for (let i = 0; i < vecA.length; i++) {
            dotProduct += vecA[i] * vecB[i];
            normA += vecA[i] ** 2;
            normB += vecB[i] ** 2;
        }

        normA = Math.sqrt(normA);
        normB = Math.sqrt(normB);

        if (normA === 0 || normB === 0) {
            return 0; // To avoid division by zero
        } else {
            return dotProduct / (normA * normB);
        }
    }

    /**
     * Take an array of chunks, and merge adjacent chunks as long as they are similar enough
     * and as long as the merged chunk doesn't exceed a token length threshold.
     * Return the merged array of chunks when done.
     */
    async mergeSplits(chunks, separator='') {
        if(!chunks || !chunks.length) {
            return [];
        }
        if (!this.tokenizer) {
            this.tokenizer = await getEncoding(this.encodingName);
        }
        const outChunks = [];
        let bar;
        if(this.showProgress) {
            bar = new SingleBar({ barsize: 80, format: '{bar} {value}/{total} chunks | {percentage}% | Time: {duration_formatted} | ETA: {eta_formatted}' }, Presets.shades_classic);
            bar.start(chunks.length, 1);
        }
        let currentChunk = chunks[0];
        for(let i=1; i<chunks.length; i++) {
            const nextChunk = chunks[i];
            const nextTokens = this.tokenizer.encode(nextChunk, this.allowedSpecial, this.disallowedSpecial);
            const currentTokens = this.tokenizer.encode(currentChunk, this.allowedSpecial, this.disallowedSpecial);

            // If too big, then move on
            if(currentTokens.length + nextTokens.length > this.chunkSize) {
                outChunks.push(currentChunk);
                currentChunk = nextChunk;
                } else {
                // If not too big, then calculate the embeddings
                const [currentEmbedding, nextEmbedding] = await this.embeddings.embedDocuments([currentChunk, nextChunk]);
                const similarity = this.cosineSimilarity(currentEmbedding, nextEmbedding);

                if(similarity > this.similarityTheshold) {
                    // If they are similar, then merge and calculate new embedding
                    currentChunk = _.join([currentChunk, nextChunk], separator);
                } else {
                    // If not similar enough then move on
                    outChunks.push(currentChunk);
                    currentChunk = nextChunk;
                }
            }
            bar.increment();
        }
        if(currentChunk) { outChunks.push(currentChunk); }
        bar.stop();
        return outChunks;
    }

    /**
     * Method that takes the documents, splits them into small chunks, calculates embedding for each chunk,
     * then assembles longer chunks up to some length limit, or a point where the similarity between chunks falls below
     * a threshold. It will then repeat this with the bigger chunks themselves calculating embeddings for the entire chunk and
     * merging adjacent similar chunks below the token limit.
     * Based on https://github.com/jparkerweb/semantic-chunking/blob/main/chunkit.js
     * @returns Promise that resolves with an array of `Document` objects.
     */
    async splitText(text) {
        if (!this.tokenizer) {
            this.tokenizer = await getEncoding(this.encodingName);
        }
        // First recursivesplit to get the initial small chunks
        const splitter = new RecursiveCharacterTextSplitter({
            chunkSize: this.initialChunkSize,
            chunkOverlap: 0,
            keepSeparator: true,
            lengthFunction: (text) => this.tokenizer.encode(text, this.allowedSpecial, this.disallowedSpecial).length,
            separators: ['\n\n', // Paragraphs
                '.', // Sentences
                '!', // Sentences
                '?', // Sentences
                '“', // Quotes
                '"', // Quotes
                ';', // Clauses
                '---', // Clauses
                '—', // Clauses [em dash]
                '--', // Clauses
                '–', // Clauses [en dash]
                ',', // Clauses
                '(', // Clauses
                ')', // Clauses
                ':', // Clauses
                '\n', // Lines
                ' ', // Words
                ''], // Anything
        });
        splitter.joinDocs = (docs, separator) => {
            const text = docs.join(separator);
            return text.trim() === "" ? null : text;
        }
        let chunks = await splitter.splitText(text);

        const chunkEmbeddings = [];
        let bar;
        if (this.showProgress) {
            bar = new SingleBar({ barsize: 80, format: '{bar} {value}/{total} embeddings | {percentage}% | Time: {duration_formatted} | ETA: {eta_formatted}' }, Presets.shades_classic);
            bar.start(chunks.length, 1);
        }
        for (let i = 0; i < chunks.length; i++) {
            chunkEmbeddings.push(await this.embeddings.embedQuery(chunks[i]));
            bar.increment();
        }
        bar.stop();

        // Now we have all the embeddings, calculate the distances to the next sentence
        const similarities = _.map(chunkEmbeddings, (emb, i) => {
            if (i == chunkEmbeddings.length - 1) { return 0; }
            return this.cosineSimilarity(emb, chunkEmbeddings[i + 1]);
        });
        const mean = _.sum(similarities) / similarities.length;
        // const stddev = Math.sqrt(_.sum(_.map(similarities, (i) => Math.pow((i - mean), 2))) / similarities.length);
        if (!this.similarityTheshold) { this.similarityTheshold = mean; }

        let numChunks = chunks.length;
        do {
            chunks = await this.mergeSplits(chunks);
            if(chunks.length == numChunks) {
                break;
            }
            numChunks = chunks.length;
        } while(true);
        return chunks;
    };
};
