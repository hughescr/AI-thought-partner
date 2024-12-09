import { DocumentInterface } from '@langchain/core/documents';
import { BaseDocumentCompressor } from '@langchain/core/retrievers/document_compressors';
import { Ollama } from 'ollama/browser';
import _ from 'lodash';

export interface OllamaRerankArgs {
    model: string
    baseUrl?: string
    topN?: number
}

export class OllamaRerank extends BaseDocumentCompressor {
    model: string;
    topN: number;
    client: Ollama;
    constructor(fields?: OllamaRerankArgs) {
        super();
        Object.defineProperty(this, 'client', {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, 'model', {
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
        this.client = new Ollama({
            host: fields?.baseUrl,
        });
        this.model = fields?.model ?? this.model;
        this.topN = fields?.topN ?? this.topN;
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
        const _docs = _.map(documents, 'pageContent');
        const { results } = await this.client.rerank({
            model: this.model,
            query,
            documents: _docs,
            top_n: this.topN ?? _docs.length,
        });
        const finalResults: DocumentInterface[] = [];
        for(const result of results) {
            const doc = _.find(documents, { pageContent: result.document });
            if(doc) {
                doc.metadata.relevanceScore = result.relevance_score;
                finalResults.push(doc);
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
        model?: string
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
        const model = options?.model ?? this.model;
        const topN = options?.topN ?? this.topN ?? docs.length;
        const { results } = await this.client.rerank({
            model,
            query,
            documents: docs,
            top_n: topN
        });
        const resultObjects = _.map(results, result => ({
            doc: result.document,
            relevanceScore: result.relevance_score,
        }));
        return resultObjects;
    }
};
export { };
