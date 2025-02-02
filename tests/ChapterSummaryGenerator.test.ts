import { describe, it, expect } from 'bun:test';
import { ChapterSummaryGenerator } from '../lib/ChapterSummaryGenerator';
import { Document } from '@langchain/core/documents';
import { RunnableLambda } from '@langchain/core/runnables';

describe('ChapterSummaryGenerator base behavior', () => {
    it('Basic summarization test', async () => {
        const dummy = new ChapterSummaryGenerator({
            targetSummarySize: 100,
            llm: RunnableLambda.from(() => ({ content: [{ text: 'Test Summary', type: 'text' }] }))
        });
        const doc = new Document({ pageContent: '# Chapter 1\nTest', metadata: { chapter: 1 } });
        const summary = await dummy.generateSummary(doc);
        expect(summary).toBeInstanceOf(Document);
        expect(summary).toMatchObject({
            pageContent: '# Chapter 1\nTest Summary',
            metadata: { chapter: 1 }
        });
    });
});
