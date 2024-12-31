import { ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { LLM } from '@langchain/core/language_models/llms';
import { Document } from '@langchain/core/documents';
import _ from 'lodash';

interface SummaryGeneratorOptions {
    llm: LLM
    targetSummarySize: number // Desired summary size in tokens
}

export class SummaryGenerator {
    private llm: LLM;
    private targetSummarySize: number;

    constructor(options: SummaryGeneratorOptions) {
        this.llm = options.llm;
        this.targetSummarySize = options.targetSummarySize;
    }

    public async generateSummary(docs: Document[]): Promise<Document> {
        const combinedText = _(docs).map('pageContent').join('\n\n');

        const summaryPrompt = ChatPromptTemplate.fromMessages([
            SystemMessagePromptTemplate.fromTemplate(`## Task
Your task is to generate a concise summary of the provided passages. Ensure that the summary does not exceed ${this.targetSummarySize} tokens.

## Guidelines
1. Read and understand all the passages carefully.
2. Focus on the main points and logical flow.
3. Ensure the summary is coherent and captures the essence of the content.
4. Output only the generated summary without any additional formatting or metadata.`),
            HumanMessagePromptTemplate.fromTemplate(`## Passages
${combinedText}

Please provide the generated summary immediately:`),
        ]);

        const summaryChain = summaryPrompt
            .pipe(this.llm)
            .pipe(new StringOutputParser());

        const summaryText = await summaryChain.invoke({});

        return new Document({
            pageContent: summaryText,
            metadata: { summary: true },
        });
    }
}
