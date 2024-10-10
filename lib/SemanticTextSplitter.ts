// Semantic chunking following the method of https://nbviewer.org/github/nesbyte/ResearchChunkingStrategies/blob/main/main.ipynb

import { TextSplitter, RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { OllamaEmbeddings } from '@langchain/ollama';
import _ from 'lodash';
import { cosineSimilarity } from './utils';
import { getEncoding } from '@langchain/core/utils/tiktoken';
import { Tiktoken, TiktokenEncoding } from 'js-tiktoken/lite';
import cliProgress from 'cli-progress';

_.mixin({
    asyncMap: async function <T, R>(collection: T[], iteratee: (value: T) => Promise<R>): Promise<R[]> {
        return Promise.all(_.map(collection, iteratee));
    }
});

interface SemanticTextSplitterOptions {
    embeddings: OllamaEmbeddings;
    chunkSize?: number;
    embeddingBatchSize?: number;
    tokenizer?: TiktokenEncoding;
    initialChunkSize?: number;
    showProgress?: boolean;
}

export class SemanticTextSplitter extends TextSplitter {
    private embeddings: OllamaEmbeddings;
    private embeddingBatchSize: number;
    private tokenizer: TiktokenEncoding;
    private initialChunkSize: number;
    private tokenizerInstance: Tiktoken;
    private showProgress: boolean;

    constructor(options: SemanticTextSplitterOptions) {
        super();
        this.embeddings = options.embeddings;
        this.chunkSize = options.chunkSize || 512;
        this.embeddingBatchSize = options.embeddingBatchSize || 16;
        this.tokenizer = options.tokenizer || 'gpt2';
        this.initialChunkSize = options.initialChunkSize || 32;
        this.showProgress = options.showProgress || false;
        this.lengthFunction = async (text: string): Promise<number> => {
            if (this.tokenizerInstance === undefined) {
                this.tokenizerInstance = await getEncoding(this.tokenizer);
            }
            return this.tokenizerInstance.encode(text).length;
        };
    }

    public async splitText(text: string): Promise<string[]> {
        const splitter: TextSplitter = new RecursiveCharacterTextSplitter({
            separators: [
                '\n\n', '.', '!', '?', '“', '"',
            ],
            keepSeparator: true,
            chunkSize: this.initialChunkSize,
            chunkOverlap: 0,
            lengthFunction: this.lengthFunction.bind(this),
        });
        splitter.splitOnSeparator = function(text: string, separator: string): string[] {
            let splits;
            if (separator) {
                if (this.keepSeparator) {
                    const regexEscapedSeparator = separator.replace(/[/\-\\^$*+?.()|[\]{}]/g, "\\$&");
                    splits = text.split(new RegExp(`(?<=${regexEscapedSeparator})`));
                }
                else {
                    splits = text.split(separator);
                }
            }
            else {
                splits = text.split("");
            }
            return splits.filter((s) => s !== "");
        };
        splitter.splitOnSeparator = splitter.splitOnSeparator.bind(splitter);

        const initialChunks: string[] = await splitter.splitText(text);
        return await this.createFinalChunks(initialChunks);
    }

    private createMergedChunks(chunks: string[]): string[] {
        return _.map(chunks.slice(0, -1), (chunk, i) => chunk + chunks[i + 1]);
    }

    private async embedMergedChunks(mergedChunks: string[]): Promise<number[][]> {
        const chunkBatches = _.chunk(mergedChunks, this.embeddingBatchSize);
        const progressBar = this.showProgress ? new cliProgress.SingleBar({ barsize: 80, format: '{bar} {value}/{total} embeddings | {percentage}% | Time: {duration_formatted} | ETA: {eta_formatted}' }, cliProgress.Presets.shades_classic) : null;

        if (this.showProgress) {
            progressBar?.start(mergedChunks.length, 0);
        }

        const embeddedChunks: number[][] = [];
        for (let i = 0; i < chunkBatches.length; i++) {
            const batch = chunkBatches[i];
            const embeddings = await this.embeddings.embedDocuments(batch);
            embeddedChunks.push(...embeddings);

            if (this.showProgress) {
                progressBar?.update(Math.min((i + 1) * this.embeddingBatchSize, mergedChunks.length));
            }
        }

        if (this.showProgress) {
            progressBar?.stop();
        }

        return embeddedChunks;
    }

    private async createFinalChunks(initialChunks: string[]): Promise<string[]> {
        const finalChunks: string[] = [];
        let currentChunk: string = '';
        const mergedChunks = this.createMergedChunks(initialChunks);
        const embeddings = await this.embedMergedChunks(mergedChunks);
        const similarities = _.map(embeddings.slice(0, -1), (embedding, i) => cosineSimilarity(embedding, embeddings[i + 1]));

        for (let i = 0; i < initialChunks.length; i++) {
            const chunk = initialChunks[i];
            const potentialChunk = currentChunk + chunk;
            const length = await this.lengthFunction(potentialChunk);

            if (length > this.chunkSize) {
                finalChunks.push(currentChunk);
                currentChunk = chunk;
            } else {
                currentChunk = potentialChunk;
            }

            if (i > 0 && i < similarities.length - 1) {
                if (similarities[i - 1] >= similarities[i] && similarities[i + 1] > similarities[i]) {
                    finalChunks.push(currentChunk);
                    currentChunk = '';
                }
            }
        }

        if (currentChunk) {
            finalChunks.push(currentChunk);
        }

        return finalChunks;
    }
}
