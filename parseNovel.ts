import {
    cachedSnowflakeArctic2Embeddings as embeddings,
    cachedJinaV2SmallENEmbeddings as fastEmbeddings,
    novaLiteLLM as summarizerLLM
} from './lib/LLMs';
import { ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { FaissStore } from '@langchain/community/vectorstores/faiss';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import cliProgress from 'cli-progress';
import { SemanticTextSplitter } from './lib/SemanticTextSplitter';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { Document } from '@langchain/core/documents';
import { SummaryGenerator } from './lib/SummaryGenerator';
import _ from 'lodash';
import { logger } from '@hughescr/logger';
import pThrottle from 'p-throttle';
import pLimit from 'p-limit';
import { getEncoding } from '@langchain/core/utils/tiktoken';
import fs from 'fs/promises';

/**
 * Saves an array of summaries to a plain text file.
 * @param summaries - Array of Document objects containing summaries.
 * @param level - The current summarization level.
 * @param book - The name of the book being processed.
 */
async function saveSummariesToFile(summaries: Document[], level: number, book: string): Promise<void> {
    const filePath = `novels/${book}_level${level}.txt`;
    const content = _.map(summaries, summary => summary.pageContent).join('\n\n');
    await fs.writeFile(filePath, content, 'utf-8');
    logger.info(`Summaries for Level ${level} saved to ${filePath}\n`);
}

_.mixin({
    awaitAll: function <T>(promiseArray: Promise<T>[]) {
        return Promise.all(promiseArray);
    },
},
{ chain: true } // Enable chaining for this mixin
);

const idealContextSize = 4096; // Desired total token limit for summaries
const book = 'DMK_V9';
// const loader = new PDFLoader(`novels/${book}.pdf`, { splitPages: true });
const loader = new TextLoader(`novels/${book}.md`);
const docs = await loader.load();

// Strategy per https://blog.getbind.co/2024/09/25/claude-contextual-retrieval-vs-rag-how-is-it-different/
// Break the source into large chunks (maybe 8k tokens semantically)
// Then, for each large chunk, break it into small chunks (maybe 256 tokens), and ask an LLM to describe the context of each small chunk (adding another 256 tokens)
// Then, concat the context and the extract, calculate encodings, and store in a vector store

class RecursiveCharacterTextSplitterSeparatorMod extends RecursiveCharacterTextSplitter {
    // Override the splitOnSeparator method to allow for keeping the separator attached to the earlier chunk not the later chunk
    // Without this, punctuation ends up on the wrong chunk... for example
    // Sentence 1. Sentence 2. ==> ['Sentence 1', '. Sentence 2', '.'] instead of ['Sentence 1.', 'Sentence 2.' ]
    // The former is clearly dumber than shit and will confuse the LLM with its weird leading periods and no end to the sentence, etc.
    splitOnSeparator(text: string, separator: string): string[] {
        let splits: string[] = [];
        if(separator) {
            if(this.keepSeparator) {
                const regexEscapedSeparator: string = _.replace(
                    separator,
                    /[/\-\\^$*+?.()|[\]{}]/g,
                    '\\$&'
                );
                splits = _.split(text, new RegExp(`(?<=${regexEscapedSeparator})`));
            } else {
                splits = _.split(text, separator);
            }
        } else {
            splits = _.split(text, '');
        }
        return _.filter(splits, s => s !== '');
    }
};

const chapterSplitter = new RecursiveCharacterTextSplitterSeparatorMod({
    separators: ['\n#', '\n\n', '.', '!', '?'], // Chapters, paragraphs, sentences
    chunkSize: 20 * 1024,
    keepSeparator: true,
    chunkOverlap: 0,
});

await chapterSplitter.splitDocuments(docs);

const targetSummarySize = idealContextSize / 10; // 3276.8 tokens if idealContextSize is 32768

const summaryGenerator = new SummaryGenerator({
    llm: summarizerLLM,
    targetSummarySize: targetSummarySize, // 400 tokens
});

const splitter: SemanticTextSplitter = new SemanticTextSplitter({
    showProgress: true,
    initialChunkSize: targetSummarySize / 4, // Tokens!
    chunkSize: targetSummarySize, // Tokens!
    embeddings: fastEmbeddings, // Use fast embeddings for decent semantic splits
    embeddingBatchSize: 128,
});

// Now go through each chapter, split it into smaller chunks, and then calculate context for each chunk.
const contextSummaryPrompt: ChatPromptTemplate = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(
        `## Task
Your task is to generate a very minimal, short explanatory context that grounds the given short passage within the context of the provided long passage. The generated context should clarify any ambiguous pronouns in the short passage by providing relevant information from the long passage. Additionally, include meta-information such as the chapter number or section where these passages are from.

## Guidelines
1. Read and understand both the long passage and the short passage carefully.
2. Identify any pronouns or ambiguous references in the short passage that require clarification from the long passage.
3. Extract the minimal necessary information from the long passage to provide context for the short passage and resolve any ambiguities.
4. Include meta-information like the chapter number or section where these passages are from.
5. Output only the generated context, without any JSON markup, quotation marks, or additional lead-in text.`
    ),
    HumanMessagePromptTemplate.fromTemplate(`## Long Passage
{long}

## Short Passage
{short}

Please provide the generated minimal explanatory context immediately:`),
]);

const contextChain = contextSummaryPrompt
    .pipe(summarizerLLM)
    .pipe(new StringOutputParser());

const limit = pLimit(32); // Limit to 32 concurrent request
const throttle = pThrottle({
    limit: 64,
    interval: 60 * 1000,
}); // Limit to 64 per minute
// We do about 4k tokens per request, and we are limited to 300,000 tokens per minute, so we can do about 75 requests per minute
const throttledInvoke = throttle(({ long, short }) => contextChain.invoke({ 'long': long, 'short': short }));
// We want to limit to a max of 32 simultaneous requests, but also throttle to 64 per minute, so combine limit and throttle:
const throttledSummaryGenerator = throttle(({ docs }) => summaryGenerator.generateSummary(docs));
const limitedThrottledSummaryGenerator = ({ docs }) => limit(() => throttledSummaryGenerator({ docs }));

// Initialize MultiBar
// Create main progress bar for chapters
// for(const chapter of chapterChunks) {
//     chapterBar.increment();
//     const smallerChunks = await splitter.splitDocuments([chapter]);

//     const totalSmallerChunks = smallerChunks.length;
//     // Create progress bar for smaller chunks
//     const chunkBar = multiBar.create(totalSmallerChunks, 0, {
//         name: 'Chunks',
//     });
//     if(_.includes(summarizerLLM.lc_namespace, 'ollama')) {
//         for(const smallerChunk of smallerChunks) {
//             const context = await contextChain.invoke({
//                 'long': chapter.pageContent,
//                 'short': smallerChunk.pageContent,
//             });
//             multiBar.log(`Context: (${context.length}) ${context}\n`);
//             chunkBar.increment();
//             smallerChunk.metadata.context = context; // Save the context in the metadata
//             splits.push(smallerChunk);
//         }
//     } else {
//         // Do the same thing, but making the calls in parallel
//         multiBar.log(chalk.yellow(`Processing ${totalSmallerChunks} smaller chunks in parallel...\n`));
//         const contexts = await Promise.all(_.map(smallerChunks, async (smallerChunk) => {
//             const context = await limitedThrottledInvoke({
//                 'long': chapter.pageContent,
//                 'short': smallerChunk.pageContent,
//             });
//             multiBar.log(`Context: (${context.length}) ${context}\n`);
//             chunkBar.increment();
//             smallerChunk.metadata.context = context; // Save the context in the metadata
//             splits.push(smallerChunk);
//             return context;
//         }));
//         multiBar.log(chalk.yellow(`Finished processing ${contexts.length} smaller chunks in parallel.\n`));
//     }

//     // Stop the smaller chunks progress bar
//     chunkBar.stop();
//     multiBar.remove(chunkBar);
// }
//
// let vectorStore: FaissStore | undefined;

// const splitChunks = _(splits)
//     .flatten()
//     .chunk(16)
//     .value();
// const bar = multiBar.create(
//     _.flatten(splits).length,
//     0,
//     { name: 'Saving chunks' }
// );
// for(const chunk of splitChunks) {
//     if(vectorStore) {
//         await vectorStore.addDocuments(chunk);
//     } else {
//         vectorStore = await FaissStore.fromDocuments(chunk, embeddings); // Index with the slower, better embeddings
//     }
//     bar.increment(chunk.length);
// }
// if(vectorStore) {
//     await vectorStore.save(`novels/${book}`);
// }

/**
 * Calculates the total number of tokens in the provided documents.
 * @param docs - Array of Document objects.
 * @returns Total token count.
 */
async function calculateTotalTokens(docs: Document[]): Promise<number> {
    let total = 0;
    const tokenizerInstance = await getEncoding('gpt2'); // Adjust tokenizer if necessary
    for(const doc of docs) {
        total += tokenizerInstance.encode(doc.pageContent).length;
    }
    return total;
}

// Initialize variables for iterative summarization
let currentSummaries = await splitter.splitDocuments(docs);
let level = 1;

while(true) {
    logger.info(`\nStarting summarization Level ${level}...\n`);

    // Calculate total tokens of current summaries
    const totalTokens = await calculateTotalTokens(currentSummaries);
    logger.info(`Total tokens at Level ${level}: ${totalTokens}`);

    // Check if total tokens are within the ideal context size
    if(totalTokens <= idealContextSize) {
        logger.info(`Desired context size achieved at Level ${level - 1}.`);
        break;
    }

    try {
        const newSummaries: Document[] = [];

        // Concatenate all summaries into a single text
        const concatenatedText = _.map(currentSummaries, doc => doc.pageContent).join('\n\n');
        const concatenatedDocument = new Document({ pageContent: concatenatedText });
        const concatenatedChunks = await splitter.splitDocuments([concatenatedDocument]);

        if(_.includes(summarizerLLM.lc_namespace, 'ollama')) {
        // Serial Processing for Ollama
            for(let i = 0; i < concatenatedChunks.length; i++) {
                const chunk = concatenatedChunks[i];
                const summary = await summaryGenerator.generateSummary([chunk]);
                newSummaries.push(summary);
                logger.info(`Generated Level ${level} summary for chunk ${i + 1}`);
            }
        } else {
        // Parallel Processing for Non-Ollama LLMs using limitedThrottledSummaryGenerator
            logger.info(`Processing ${concatenatedChunks.length} chunks in parallel...\n`);

            const batchSize = 10; // Define an appropriate batch size within each level
            const batchPromises: Promise<Document | null>[] = [];

            for(let i = 0; i < concatenatedChunks.length; i += batchSize) {
                const batch = concatenatedChunks.slice(i, i + batchSize);
                const promise = limitedThrottledSummaryGenerator({ docs: batch })
                .then((summary, levelCopy = level) => {
                    logger.info(`Generated Level ${level} summary for batches ${i + 1} to ${i + batch.length}`);
                    return summary;
                })
                .catch((error) => {
                    logger.info(`Error generating summary for batches ${i + 1} to ${i + batch.length}: ${error.message}`);
                    return null; // Handle error by returning null or appropriate placeholder
                });
                batchPromises.push(promise);
            }

            const summaries = await Promise.all(batchPromises);

            // Filter out any null summaries due to errors
            const successfulSummaries = _.filter(summaries, summary => summary !== null) as Document[];

            newSummaries.push(...successfulSummaries);

            logger.info(`Finished processing ${successfulSummaries.length} summaries in parallel.\n`);
        }

        // Save the summaries to a plain text file
        await saveSummariesToFile(newSummaries, level, book);

        // Optionally, index the new summaries into FaissStore (Uncomment if indexing is still needed)
        logger.info(`Indexing Level ${level} summaries...\n`);
        const vectorStoreLevel = await FaissStore.fromDocuments(newSummaries, embeddings);
        await vectorStoreLevel.save(`novels/${book}_level${level}`);
        logger.info(`Level ${level} summaries indexed and saved.\n`);

        // Prepare for next iteration
        // Concatenate all new summaries for the next level's input
        const concatenatedNewSummaries = _.map(newSummaries, doc => doc.pageContent).join('\n\n');
        const nextDocuments = await splitter.splitDocuments([new Document({ pageContent: concatenatedNewSummaries })]);
        currentSummaries = nextDocuments;
        level++;
    } catch(error) {
        logger.info(`Error during Level ${level} summarization or FaissStore indexing: ${error.message}`);
        break; // Exit loop on error
    }
}
