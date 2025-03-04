import { describe, it, expect, beforeAll, afterAll, jest } from 'bun:test';
import { FaissStoreWithMMR } from '../lib/FAISSStoreWithMMR';
import { Document } from '@langchain/core/documents';
import _ from 'lodash';
// mathUtils import removed as it's not used
import type { Embeddings } from '@langchain/core/embeddings';
import type { AsyncCaller } from '@langchain/core/utils/async_caller';
import { cosineSimilarity } from '../lib/utils';
import { cachedSnowflakeArctic2Embeddings } from '../lib/LLMs';

// Real test documents with consistent patterns for real embeddings test
const realTestDocuments = [
    new Document({ pageContent: 'Cats are domestic animals that are independent and like to sleep a lot.', metadata: { topic: 'animals/cats' } }),
    new Document({ pageContent: 'Dogs are domestic animals that are loyal and enjoy playing with their owners.', metadata: { topic: 'animals/dogs' } }),
    new Document({ pageContent: 'Parrots are colorful birds that can mimic human speech and are very intelligent.', metadata: { topic: 'animals/birds' } }),
    new Document({ pageContent: 'JavaScript is a programming language used primarily for web development.', metadata: { topic: 'technology/programming' } }),
    new Document({ pageContent: 'Python is a popular programming language known for its readability and versatility.', metadata: { topic: 'technology/programming' } }),
    new Document({ pageContent: 'Machine learning is a subset of artificial intelligence focused on data and algorithms.', metadata: { topic: 'technology/ai' } }),
    new Document({ pageContent: 'The Roman Empire was one of the largest empires in ancient history.', metadata: { topic: 'history/ancient' } }),
    new Document({ pageContent: 'World War II was a global conflict that lasted from 1939 to 1945.', metadata: { topic: 'history/modern' } })
];

// Tests with minimal mocking and real diversity validation
describe('FaissStoreWithMMR - Diversity Validation', () => {
    let mmrStore: FaissStoreWithMMR;
    let docEmbeddings: number[][];

    // Helper function to calculate average similarity between documents
    const calculateAverageSimilarity = (docs: Document[]): number => {
        if(docs.length <= 1) {
            return 1;
        }

        let totalSimilarity = 0;
        let pairCount = 0;

        // For each pair of documents
        for(let i = 0; i < docs.length; i++) {
            for(let j = i + 1; j < docs.length; j++) {
                // Get the doc contents for lookup in our docEmbeddings
                const docI = docs[i].pageContent;
                const docJ = docs[j].pageContent;

                // Find the index of the docs in our original array
                const indexI = _.findIndex(realTestDocuments, ['pageContent', docI]);
                const indexJ = _.findIndex(realTestDocuments, ['pageContent', docJ]);

                if(indexI >= 0 && indexJ >= 0) {
                    // Calculate cosine similarity between embeddings
                    const similarity = cosineSimilarity(docEmbeddings[indexI], docEmbeddings[indexJ]);
                    totalSimilarity += similarity;
                    pairCount++;
                }
            }
        }

        return pairCount > 0 ? totalSimilarity / pairCount : 0;
    };

    // Create a FaissStore with the real test documents
    beforeAll(async () => {
        // Generate synthetic embeddings that reflect the semantic relationships
        // For testing we need embeddings with these properties:
        // - Documents about the same topic should have higher similarity
        // - Documents about different topics should have lower similarity

        docEmbeddings = [
            // Animals - cat
            [0.9, 0.8, 0.7, 0.3, 0.2, 0.1, 0.0, 0.0],
            // Animals - dog
            [0.85, 0.9, 0.6, 0.3, 0.2, 0.1, 0.0, 0.0],
            // Animals - bird
            [0.8, 0.7, 0.9, 0.2, 0.1, 0.1, 0.0, 0.0],
            // Tech - JavaScript
            [0.1, 0.2, 0.1, 0.9, 0.8, 0.6, 0.1, 0.0],
            // Tech - Python
            [0.1, 0.2, 0.1, 0.8, 0.9, 0.7, 0.1, 0.0],
            // Tech - ML/AI
            [0.2, 0.1, 0.2, 0.7, 0.7, 0.9, 0.1, 0.0],
            // History - Roman
            [0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.9, 0.7],
            // History - WW2
            [0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.8, 0.9]
        ];

        // Create synthetic query embeddings
        const queryEmbeddings = {
            cats: [0.95, 0.7, 0.6, 0.1, 0.1, 0.0, 0.0, 0.0],
            animals: [0.85, 0.8, 0.8, 0.2, 0.1, 0.1, 0.0, 0.0],
            programming: [0.1, 0.1, 0.1, 0.9, 0.8, 0.5, 0.0, 0.0]
        };

        // Use a real FaissStore, but with controlled embeddings
        const mockRealEmbeddings: Embeddings = {
            embedQuery: jest.fn((query: string) => {
                // Return our synthetic query embeddings based on the query
                if(query.includes('cat')) {
                    return Promise.resolve(queryEmbeddings.cats);
                }
                if(query.includes('animal')) {
                    return Promise.resolve(queryEmbeddings.animals);
                }
                if(query.includes('programming')) {
                    return Promise.resolve(queryEmbeddings.programming);
                }
                // Default to animals query
                return Promise.resolve(queryEmbeddings.animals);
            }),
            embedDocuments: jest.fn((_texts: string[]) => {
                // Return our pre-generated document embeddings
                return Promise.resolve(docEmbeddings);
            }),
            caller: {} as AsyncCaller
        };

        // Create the MMR store with our controlled embeddings
        mmrStore = new FaissStoreWithMMR(mockRealEmbeddings, {});

        // Define mock function
        const mockSimilaritySearch = jest.fn(async (queryVector: number[], fetchK: number) => {
            // For animal queries
            if(queryVector[0] > 0.8) {
                // Calculate similarities with all documents
                const similarities = _.map(docEmbeddings, embedding => cosineSimilarity(queryVector, embedding));

                // Create pairs of [document, score]
                const pairs = _.map(realTestDocuments, (doc, i) => [doc, similarities[i]]);

                // Sort by similarity score (descending)
                const sorted = _.sortBy(pairs, pair => -pair[1]);

                // Return top fetchK results
                return sorted.slice(0, fetchK);
            }

            // For programming queries
            if(queryVector[3] > 0.8) {
                const techDocs = [
                    [realTestDocuments[3], 0.95], // JavaScript
                    [realTestDocuments[4], 0.90], // Python
                    [realTestDocuments[5], 0.85], // ML/AI
                    [realTestDocuments[0], 0.2],  // Cats (less relevant)
                    [realTestDocuments[1], 0.15], // Dogs (less relevant)
                ];
                return techDocs.slice(0, fetchK);
            }

            // Default to returning all docs with some similarity
            return _.map(realTestDocuments, (doc, i) => [doc, 0.9 - (i * 0.1)]).slice(0, fetchK);
        });

        // Apply the mock to the instance method
        mmrStore.similaritySearchVectorWithScore = mockSimilaritySearch;
    });

    afterAll(() => {
        jest.restoreAllMocks();
    });

    it('produces more diverse results with lower lambda values', async () => {
        // Query related to animals - should match first few documents best
        const query = 'Tell me about animals';

        // Try different lambda values
        const highDiversityResults = await mmrStore.maxMarginalRelevanceSearch(query, { k: 4, lambda: 0.1 });
        const mediumDiversityResults = await mmrStore.maxMarginalRelevanceSearch(query, { k: 4, lambda: 0.5 });
        const lowDiversityResults = await mmrStore.maxMarginalRelevanceSearch(query, { k: 4, lambda: 0.9 });

        // Calculate average similarity between documents in each result set
        const highDiversitySimilarity = calculateAverageSimilarity(highDiversityResults);
        const mediumDiversitySimilarity = calculateAverageSimilarity(mediumDiversityResults);
        const lowDiversitySimilarity = calculateAverageSimilarity(lowDiversityResults);

        // Verify that lower lambda produces more diverse results (lower average similarity)
        expect(highDiversitySimilarity).toBeLessThan(lowDiversitySimilarity);
        expect(mediumDiversitySimilarity).toBeLessThan(lowDiversitySimilarity);

        // Count unique topics in each result set
        const getUniqueTopics = (docs: Document[]): Set<string> => {
            return new Set(_.map(docs, doc => doc.metadata.topic as string));
        };

        const highDiversityTopics = getUniqueTopics(highDiversityResults);
        const lowDiversityTopics = getUniqueTopics(lowDiversityResults);

        // Expect more diverse topics with lower lambda
        expect(highDiversityTopics.size).toBeGreaterThanOrEqual(lowDiversityTopics.size);
    });
});

// Test with real embeddings from LLMs.ts
describe('FaissStoreWithMMR - Real Embeddings', () => {
    let faissStoreWithRealEmbeddings: FaissStoreWithMMR;

    // Test with a local embedding model
    it('works with real embedding model', async () => {
        // Initialize store with real embeddings
        faissStoreWithRealEmbeddings = new FaissStoreWithMMR(
            cachedSnowflakeArctic2Embeddings,
            {}
        );

        // Add our test documents
        await faissStoreWithRealEmbeddings.addDocuments(realTestDocuments);

        // Test queries with different lambda values
        const highDiversityResults = await faissStoreWithRealEmbeddings.maxMarginalRelevanceSearch(
            'Tell me about animals',
            { k: 4, lambda: 0.1 }
        );

        const lowDiversityResults = await faissStoreWithRealEmbeddings.maxMarginalRelevanceSearch(
            'Tell me about animals',
            { k: 4, lambda: 0.9 }
        );

        // Check that we got results
        expect(highDiversityResults.length).toBe(4);
        expect(lowDiversityResults.length).toBe(4);

        // If we had an animal query, low diversity should have more animals
        const animalTopicCount = (docs: Document[]): number => {
            return _.filter(docs, doc => _.startsWith(doc.metadata.topic as string, 'animals')).length;
        };

        // With a low lambda (high diversity), we expect fewer animal topics and more variety
        const highDiversityAnimalCount = animalTopicCount(highDiversityResults);
        const lowDiversityAnimalCount = animalTopicCount(lowDiversityResults);

        // If we have enough documents, diversity should be evident here
        if(lowDiversityAnimalCount > 1) {
            expect(highDiversityAnimalCount).toBeLessThanOrEqual(lowDiversityAnimalCount);
        }

        // Count unique topic categories
        const uniqueTopicCategories = (docs: Document[]): Set<string> => {
            return new Set(_.map(docs, (doc) => {
                const topic = doc.metadata.topic as string;
                return _.split(topic, '/')[0]; // Get the main category
            }));
        };

        const highDiversityCategories = uniqueTopicCategories(highDiversityResults);
        const lowDiversityCategories = uniqueTopicCategories(lowDiversityResults);

        // High diversity should have more categories
        expect(highDiversityCategories.size).toBeGreaterThanOrEqual(lowDiversityCategories.size);
    });
});
