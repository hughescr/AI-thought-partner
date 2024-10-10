import _ from 'lodash';

/**
 * Calculates the cosine similarity between two vectors.
 *
 * @param vectorA - The first vector.
 * @param vectorB - The second vector.
 * @returns The cosine similarity as a number between -1 and 1, or 0 if either vector has a norm of 0.
 */
export function cosineSimilarity(vectorA: number[], vectorB: number[]): number {
    const dotProduct: number = _.sum(_.map(vectorA, (val, index) => val * vectorB[index]));
    const normA: number = Math.sqrt(_.sum(_.map(vectorA, (val) => val * val)));
    const normB: number = Math.sqrt(_.sum(_.map(vectorB, (val) => val * val)));

    if (normA === 0 || normB === 0) {
        return 0;
    }

    return dotProduct / (normA * normB);
}
