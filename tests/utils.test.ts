import { cosineSimilarity } from '../lib/utils';

test('cosineSimilarity should return 1 for identical vectors', () => {
    const vectorA = [1, 2, 3];
    const vectorB = [1, 2, 3];
    const similarity = cosineSimilarity(vectorA, vectorB);
    expect(similarity).toBeCloseTo(1);
});

test('cosineSimilarity should return 0 for orthogonal vectors', () => {
    const vectorA = [1, 0, 0];
    const vectorB = [0, 1, 0];
    const similarity = cosineSimilarity(vectorA, vectorB);
    expect(similarity).toBeCloseTo(0);
});

test('cosineSimilarity should handle zero vectors', () => {
    const vectorA = [0, 0, 0];
    const vectorB = [0, 0, 0];
    const similarity = cosineSimilarity(vectorA, vectorB);
    expect(similarity).toBe(0);
});

test('cosineSimilarity should return correct value for non-trivial vectors', () => {
    const vectorA = [1, 2, 3];
    const vectorB = [4, 5, 6];
    const similarity = cosineSimilarity(vectorA, vectorB);
    expect(similarity).toBeCloseTo(0.9746318461970762);
});
