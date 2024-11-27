import { FaissStore } from '@langchain/community/vectorstores/faiss';
import { maximalMarginalRelevance } from '@langchain/core/utils/math';
import { MaxMarginalRelevanceSearchOptions } from '@langchain/core/vectorstores';
import _ from 'lodash';

/**
 * Return documents selected using the maximal marginal relevance.
 * Maximal marginal relevance optimizes for similarity to the query AND diversity
 * among selected documents.
 *
 * @param {string} query - Text to look up documents similar to.
 * @param {number} options.k - Number of documents to return.
 * @param {number} options.fetchK=20- Number of documents to fetch before passing to the MMR algorithm.
 * @param {number} options.lambda=0.5 - Number between 0 and 1 that determines the degree of diversity among the results,
 *                 where 0 corresponds to maximum diversity and 1 to minimum diversity.
 * @param {any} options.filter - filter parameter is ignored for FAISS stores.
 *
 * @returns {Promise<Document[]>} - List of documents selected by maximal marginal relevance.
 */
export class FaissStoreWithMMR extends FaissStore {
    async maxMarginalRelevanceSearch(query: string, options: MaxMarginalRelevanceSearchOptions<this['FilterType']>, _callbacks?: undefined) {
        const { k, fetchK = 20, lambda = 0.5 } = options;
        const queryEmbedding = await this.embeddings.embedQuery(query);
        const resultDocs = await this.similaritySearchVectorWithScore(queryEmbedding, fetchK);
        const embeddingList = await this.embeddings.embedDocuments(_.map(resultDocs, '0.pageContent'));
        const mmrIndexes = maximalMarginalRelevance(queryEmbedding, embeddingList, lambda, k);
        return _.map(mmrIndexes, idx => resultDocs[idx][0]);
    }
};
