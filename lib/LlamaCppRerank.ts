import type { DocumentInterface } from '@langchain/core/documents';
import { BaseDocumentCompressor } from '@langchain/core/retrievers/document_compressors';
import { getLlama, Llama, LlamaModel, LlamaRankingContext } from 'node-llama-cpp';
import _ from 'lodash';

export interface LlamaCppRerankArgs {
    modelPath: string
    topN?: number
}

export class LlamaCppRerank extends BaseDocumentCompressor {
    llama!: Llama;
    model!: LlamaModel;
    context!: LlamaRankingContext;
    topN: number | undefined;
    initializationPromise!: Promise<void>;

    constructor(fields: LlamaCppRerankArgs) {
        super();
        Object.defineProperty(this, 'model', {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, 'modelPath', {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, 'topN', {
            enumerable: true,
            configurable: true,
            writable: true,
            value: 3
        });
        this.topN = fields.topN;
        // Set up a mutex-like promise that resolves when initialization is complete
        this.initializationPromise = (async () => {
            this.llama = await getLlama();
            this.model = await this.llama.loadModel({ modelPath: fields.modelPath });
            this.context = await this.model.createRankingContext();
        })();
    }

    /**
     * Compress documents using Ollama's rerank API.
     *
     * @param {Array<DocumentInterface>} documents A sequence of documents to compress.
     * @param {string} query The query to use for compressing the documents.
     *
     * @returns {Promise<Array<DocumentInterface>>} A sequence of compressed documents.
     */
    async compressDocuments(documents: DocumentInterface[], query: string): Promise<DocumentInterface[]> {
        const docs = _.map(documents, 'pageContent');
        // Wait for initialization to complete.
        await this.initializationPromise;
        const rankedDocs = await this.context.rankAndSort(query, docs);
        const finalResults: DocumentInterface[] = [];
        for(const rankedDoc of rankedDocs) {
            const doc = _.find(documents, { pageContent: rankedDoc.document });
            if(doc) {
                if(!doc.metadata) {
                    doc.metadata = {};
                }
                doc.metadata.relevanceScore = rankedDoc.score;
                finalResults.push(doc);
            }
            if(this.topN && finalResults.length >= this.topN) {
                break;
            }
        }
        return finalResults;
    }

    /**
     * Returns an ordered list of documents ordered by their relevance to the provided query.
     *
     * @param {Array<DocumentInterface | string | Record<string, string>>} documents A list of documents as strings, DocumentInterfaces or objects with a `pageContent` key.
     * @param {string} query The query to use for reranking the documents.
     * @param {string} options.model The name of the model to use.
     * @param {number} options.topN How many documents to return. Default is all documents, sorted by relevance but not filtered.
     *
     * @returns {Promise<Array<{ index: number; relevanceScore: number }>>} An ordered list of documents with relevance scores.
     */
    async rerank(documents: (DocumentInterface | string | Record<string, string>)[], query: string, options?: {
        topN?: number
    }): Promise<{
            doc: string
            relevanceScore: number
        }[]> {
        const docs = _.map(documents, (doc) => {
            if(_.isString(doc)) {
                return doc;
            }
            return doc.pageContent;
        });
        const topN = options?.topN ?? this.topN ?? docs.length;
        // Wait for initialization to complete.
        await this.initializationPromise;
        const rankedDocs = await this.context.rankAndSort(query, docs);
        const resultObjects = _(rankedDocs).slice(0, topN).map(result => ({
            doc: result.document,
            relevanceScore: result.score,
        })).value();
        return resultObjects;
    }
};
export { };
