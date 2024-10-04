import _ from 'lodash';
_.mixin({
    awaitAll: function(promiseArray) {
        return Promise.all(promiseArray);
    }
}, { chain: false });

import { SingleBar, Presets } from 'cli-progress';

import { TextSplitter, RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { getEncoding } from '@langchain/core/utils/tiktoken';

/**
 * This class first uses a RecursiveCharacterTextSplitter to split the documents into tiny chunks, then re-merges
 * those chunks based on semantic similarity. It will stop when chunks reach the maximum token length or when
 * the similarity drops below some threshold, leaving a set of chunks which have internal semantic similarity
 */
export class SemanticTextSplitter extends TextSplitter {
    static lc_name = _.constant('SemanticTextSplitter');
    constructor(fields) {
        super({ chunkSize: 1000, ...fields, chunkOverlap: 0 });
        Object.defineProperty(this, 'showProgress', {
            enumerable: true,
            configurable: true,
            writable: true,
            value: false,
        });
        Object.defineProperty(this, 'embeddings', {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0,
        });
        Object.defineProperty(this, 'embeddingsBatchSize', {
            enumerable: true,
            configurable: true,
            writable: true,
            value: 16,
        });
        Object.defineProperty(this, 'tokenizer', {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0,
        });
        Object.defineProperty(this, 'encodingName', {
            enumerable: true,
            configurable: true,
            writable: true,
            value: 'gpt2',
        });
        Object.defineProperty(this, 'allowedSpecial', {
            enumerable: true,
            configurable: true,
            writable: true,
            value: [],
        });
        Object.defineProperty(this, 'disallowedSpecial', {
            enumerable: true,
            configurable: true,
            writable: true,
            value: 'all',
        });
        Object.defineProperty(this, 'initialChunkSize', {
            enumerable: true,
            configurable: true,
            writable: true,
            value: 50,
        });
        this.showProgress = fields?.showProgress ?? false;
        this.embeddings = fields?.embeddings;
        this.embeddingsBatchSize = fields?.embeddingsBatchSize ?? 16;
        this.initialChunkSize = fields?.initialChunkSize ?? 50;
        this.encodingName = fields?.encodingName ?? 'gpt2';
        this.allowedSpecial = fields?.allowedSpecial ?? [];
        this.disallowedSpecial = fields?.disallowedSpecial ?? 'all';
    }

    /**
     * Calculates the cosine similarity between two vectors (vecA and vecB).
     * @param {Array} vecA - The first vector as an array of numeric values.
     * @param {Array} vecB - The second vector as an array of numeric values.
     * @returns {number} Float value representing the cosine similarity between vecA and vecB, or 0 if any component vector has a norm of 0.
     */
    cosineSimilarity(vecA, vecB) {
        const results = _(vecA)
                        .zip(vecB)
                        .reduce((result, pair) => {
                            const [val1, val2] = pair;
                            const dotProduct = result.dotProduct + (val1 * val2);
                            const normA = result.normA + (val1 ** 2);
                            const normB = result.normB + (val2 ** 2);

                            return {
                                dotProduct: dotProduct,
                                normA: normA,
                                normB: normB,
                            };
                        }, {
                            dotProduct: 0.0,
                            normA: 0.0,
                            normB: 0.0,
                        });
        if(results.normA === 0.0 || results.normB === 0.0) { // Avoid ÷0
            return 0;
        }
        return results.dotProduct / Math.sqrt(results.normA * results.normB);
    }

    /**
     * Take an array of chunks, and merge adjacent chunks as long as they are similar enough
     * and as long as the merged chunk doesn't exceed a token length threshold.
     * Return the merged array of chunks when done.
     */
    async mergeSplits(chunks, separator = '') {
        if(!chunks || !chunks.length) {
            return [];
        }
        if(!this.tokenizer) {
            this.tokenizer = await getEncoding(this.encodingName);
        }
        const outChunks = [];
        let bar;

        // Buffer chunks behind this chunk to smooth spikes
        const smoothedChunks = this.mergeChunks(chunks, 3);

        // Get embeddings for each smoothed chunk
        if(this.showProgress) {
            bar = new SingleBar({ barsize: 80, format: '{bar} {value}/{total} embeddings | {percentage}% | Time: {duration_formatted} | ETA: {eta_formatted}' }, Presets.shades_classic);
            bar.start(chunks.length, 0);
        }
        const chunkEmbeddings = await _(smoothedChunks)
                                .chunk(this.embeddingsBatchSize)
                                .map(batch => this.embeddings.embedDocuments(batch).then((embeddings) => {
                                    bar.increment(embeddings.length);
                                    return embeddings;
                                }))
                                .awaitAll();
        bar.stop();

        // Now we have all the embeddings, calculate the distances to the next sentence
        const distances = _.map(chunkEmbeddings, (emb, i) => {
            if(i == chunkEmbeddings.length - 1) {
                return 0;
            }
            const similarity = this.cosineSimilarity(emb, chunkEmbeddings[i + 1]);
            return 1 - similarity;
        });
        const mean = _.sum(distances) / distances.length;
        const stddev = Math.sqrt(_(distances).map(i => Math.pow((i - mean), 2)).sum() / distances.length);
        const distanceThreshold = mean + stddev; // Roughly the 84th percentile

        // We want to merge chunks as long as the threshold has not been reached and the token limit has not been exceeded

        // Tokenize each chunk
        const chunkTokens = _.map(chunks, chunk => this.tokenizer.encode(chunk, this.allowedSpecial, this.disallowedSpecial));

        let currentChunk = chunks[0];
        let currentLength = chunkTokens[0].length;
        for(let i = 1; i < chunks.length; i++) {
            const nextChunk = chunks[i];
            const nextLength = chunkTokens[i].length;

            // If too big, then we can't merge so move onwards; if distance is too big then stop merge and move onwards
            if(((currentLength + nextLength) > this.chunkSize) || (distances[i - 1] > distanceThreshold)) {
                outChunks.push(currentChunk);
                currentChunk = nextChunk;
                currentLength = nextLength;
            } else {
                // If they are similar, then merge and continue
                currentChunk = _.join([currentChunk, nextChunk], separator);
                currentLength += nextLength;
            }
            bar.increment();
        }
        if(currentChunk) {
            outChunks.push(currentChunk);
        }
        bar.stop();
        return outChunks;
    }

    mergeChunks(chunks, bufferSize = 1) {
        return _.map(chunks, (chunk, i) => _.join(chunks.slice(i, i + bufferSize + 1)));
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
        if(!this.tokenizer) {
            this.tokenizer = await getEncoding(this.encodingName);
        }
        // First recursivesplit to get the initial small chunks
        const splitter = new RecursiveCharacterTextSplitter({
            chunkSize: this.initialChunkSize,
            chunkOverlap: 0,
            keepSeparator: true,
            lengthFunction: text => this.tokenizer.encode(text, this.allowedSpecial, this.disallowedSpecial).length,
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
                '\u2013', // Clauses [en dash]
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
            return _.trim(text) === '' ? null : text;
        };
        splitter.splitOnSeparator = (text, separator) => {
            let splits;
            if(separator) {
                if(splitter.keepSeparator) {
                    const regexEscapedSeparator = _.replace(separator, /[/\-\\^$*+?.()|[\]{}]/g, '\\$&');
                    splits = _.split(text, new RegExp(`(?<=${regexEscapedSeparator})`)); // <<<=== keep the separator on the PREVIOUS chunk
                } else {
                    splits = _.split(text, separator);
                }
            } else {
                splits = _.split(text, '');
            }
            return _.filter(splits, s => s !== '');
        };
        let chunks = await splitter.splitText(text);

        let numChunks = chunks.length;
        while(true) {
            chunks = await this.mergeSplits(chunks);
            if(chunks.length == numChunks) {
                break;
            }
            numChunks = chunks.length;
        }
        return chunks;
    };
};
