import { mock, jest } from 'bun:test';
import type { DocumentInterface } from '@langchain/core/documents';
import _ from 'lodash';

mock.module('ollama/browser', () => {
    return {
        Ollama: class {
            rerank = jest.fn().mockImplementation(({ model, documents }) => {
                if(model !== 'test-model') {
                    throw new Error(`model "${model}" not found, try pulling it first`);
                }
                const results = _.map(documents, (doc: DocumentInterface, index) => ({
                    document: doc,
                    relevance_score: parseFloat((0.9 * Math.pow(0.9, index)).toFixed(3)),
                }));
                return Promise.resolve({ results });
            });
        },
    };
});
