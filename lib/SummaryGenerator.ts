import { ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { Runnable } from '@langchain/core/runnables';
import { Document } from '@langchain/core/documents';
import _ from 'lodash';

interface SummaryGeneratorOptions {
    llm: Runnable
    targetSummarySize: number // Desired summary size in tokens
}

const summaryPrompt = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(`## Task
Your task is to generate a concise summary of the provided passages. Ensure that the summary does not exceed {targetSummarySize} tokens.

## Guidelines
1. Read and understand all the passages carefully.
2. Focus on the main narrative points, characters, and the plot flow of the story.
3. Ensure the summary is coherent and captures the essence of the content.
4. Output only the generated summary without any additional formatting or metadata.
5. Just provide the summary, do not lead in with "The passages say that..." or "In these passages..." or "The story is about...", etc. Just the raw summary.
6. Make sure that the summary captures all the most important narrative elements from the passages.`),
    HumanMessagePromptTemplate.fromTemplate(`## Passages
{combinedText}

Please provide the generated summary immediately:`),
]);

export class SummaryGenerator {
    private targetSummarySize: number;
    private summaryChain: Runnable;

    constructor(options: SummaryGeneratorOptions) {
        this.summaryChain = summaryPrompt.pipe(options.llm).pipe(new StringOutputParser());
        this.targetSummarySize = options.targetSummarySize;
    }

    public async generateSummary(docs: Document[]): Promise<Document> {
        const combinedText = _(docs).map('pageContent').join('');

        const summaryText = await this.summaryChain.invoke({
            targetSummarySize: this.targetSummarySize,
            combinedText
        });

        return new Document({
            pageContent: summaryText,
            metadata: { summary: true },
        });
    }
}
