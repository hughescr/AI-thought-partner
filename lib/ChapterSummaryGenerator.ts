import { ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { Runnable } from '@langchain/core/runnables';
import { Document } from '@langchain/core/documents';
import _ from 'lodash';

export interface SummaryGeneratorOptions {
    llm: Runnable
    targetSummarySize: number // Desired summary size in tokens
}

const summaryPrompt = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(`You are an expert literary analyst. Your primary goal is to generate accurate, concise summaries based solely on provided text.
Follow these guidelines for every response:
1. Never include details not in the source material.
2. Avoid meta-statements about summarizing or referencing the prompt.
3. Keep summaries under {targetSummarySize} tokens.
4. Maintain a neutral, third-person tone.
5. If you approach the token limit, omit minor details first and prioritize essential plot/character points.
6. Do not quote verbatim from the text; paraphrase as needed.

*CRITICAL*: do not make things up about the text. Only summarize what is in the text. Verify that all characters in the summary are in the text. If a character is not in the text, do not include them in the summary. Same for plot points, settings, and any other facts. DO NOT MAKE UP FACTS.

If the text does not seem to have anything happening in it, summarize it as such. Do not add details that are not in the text.
Some sections of text might just be metadata or other non-story content.
If the text does not appear to be part of the story, then just say "not part of the story".
`),
    HumanMessagePromptTemplate.fromTemplate(`Summarize the following chapter text in one concise paragraph. Focus on critical plot points, characters, and events.
No extra commentary, no headings, and no references to the prompt itself.

{chapterText}

Now provide the summary immediately (no intros, no disclaimers):`),
]);

export class ChapterSummaryGenerator {
    private targetSummarySize: number;
    private summaryChain: Runnable;

    constructor(options: SummaryGeneratorOptions) {
        this.summaryChain = summaryPrompt.pipe(options.llm).pipe(new StringOutputParser());
        this.targetSummarySize = options.targetSummarySize;
    }

    public async generateSummary(chapter: Document): Promise<Document> {
        const chapterHeader = _.split(chapter.pageContent, '\n')[0];
        if(!_.startsWith(chapterHeader, '#')) {
            throw new Error('Chapter must start with a chapter heading beginning with "#" eg "# Chapter 1: The Beginning"');
        }
        // Remove the header from the text
        const chapterText = _.replace(chapter.pageContent, chapterHeader, '');

        const summaryText = await this.summaryChain.invoke({
            targetSummarySize: this.targetSummarySize,
            chapterText: chapterText,
        });

        return new Document({
            // Prepend the chapter header back onto the summary after generation.
            pageContent: _.join([chapterHeader, summaryText], '\n'),
            metadata: { summary: true, ...chapter.metadata },
        });
    }
}
