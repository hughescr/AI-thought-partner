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
import chalk from 'chalk';
import pThrottle from 'p-throttle';
import pLimit from 'p-limit';
import { getEncoding } from '@langchain/core/utils/tiktoken';

_.mixin({
    awaitAll: function <T>(promiseArray: Promise<T>[]) {
        return Promise.all(promiseArray);
    },
},
{ chain: true } // Enable chaining for this mixin
);

const idealContextSize = 32768; // Desired total token limit for summaries
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

const chapterChunks = await chapterSplitter.splitDocuments(docs);

const targetSummarySize = idealContextSize / 10; // 3276.8 tokens if idealContextSize is 32768

const summaryGenerator = new SummaryGenerator({
    llm: summarizerLLM,
    targetSummarySize: targetSummarySize, // 400 tokens
});

const splitter: SemanticTextSplitter = new SemanticTextSplitter({
    showProgress: false,
    initialChunkSize: 32, // Tokens!
    chunkSize: 512, // Tokens!
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
}); // Limit to 300 per minute
// We do about 4k tokens per request, and we are limited to 300,000 tokens per minute, so we can do about 75 requests per minute
// We want to limit to a max of 32 simultaneous requests, but also throttle to 300 per minute, so combine limit and throttle:
const throttledInvoke = throttle(({ long, short }) => contextChain.invoke({ long, short }));
const limitedThrottledInvoke = ({ long, short }) => limit(() => throttledInvoke({ long, short }));

// Initialize MultiBar
const multiBar = new cliProgress.MultiBar(
    {
        clearOnComplete: false,
        hideCursor: true,
        format:
            '{bar} {value}/{total} {name} | {percentage}% | Time: {duration_formatted} | ETA: {eta_formatted}',
    },
    cliProgress.Presets.shades_classic
);
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
    multiBar.log(chalk.blue(`\nStarting summarization Level ${level}...\n`));

    // Calculate total tokens of current summaries
    const totalTokens = await calculateTotalTokens(currentSummaries);
    multiBar.log(chalk.blue(`Total tokens at Level ${level}: ${totalTokens}`));

    // Check if total tokens are within the ideal context size
    if(totalTokens <= idealContextSize) {
        multiBar.log(chalk.green(`Desired context size achieved at Level ${level - 1}.`));
        break;
    }

    try {
        const newSummaries: Document[] = [];

        if(_.includes(summarizerLLM.lc_namespace, 'ollama')) {
            // Serial Processing for Ollama
            for(let i = 0; i < currentSummaries.length; i += 10) {
                const batch = currentSummaries.slice(i, i + 10);
                const summary = await summaryGenerator.generateSummary(batch);
                newSummaries.push(summary);
                multiBar.log(chalk.green(`Generated Level ${level} summary for batches ${i + 1} to ${i + batch.length}`));
            }
        } else {
            // Parallel Processing for Non-Ollama LLMs
            multiBar.log(chalk.yellow(`Processing ${currentSummaries.length} summaries in parallel...\n`));

            const batchSize = 10; // Define an appropriate batch size
            const batchPromises: Promise<void>[] = [];

            for(let i = 0; i < currentSummaries.length; i += batchSize) {
                const batch = currentSummaries.slice(i, i + batchSize);
                const promise = limitedThrottledInvoke({
                    long: _.map(batch, doc => doc.pageContent).join('\n'), // Assuming concatenation; adjust as needed
                    short: ' ' // Provide appropriate short passage if required
                }).then((summary) => {
                    newSummaries.push(summary);
                    multiBar.log(chalk.green(`Generated Level ${level} summary for batch ${i + 1} to ${i + batch.length}`));
                    return summary; // Added return statement
                }).catch(error => {
                    multiBar.log(chalk.red(`Error generating summary for batch ${i + 1} to ${i + batch.length}: ${error.message}`));
                });
                batchPromises.push(promise);
            }

            await Promise.all(batchPromises);

            multiBar.log(chalk.yellow(`Finished processing ${currentSummaries.length} summaries in parallel.\n`));
        }

        // Index the new summaries into FaissStore
        multiBar.log(chalk.blue(`Indexing Level ${level} summaries...\n`));
        const vectorStoreLevel = await FaissStore.fromDocuments(newSummaries, embeddings);
        await vectorStoreLevel.save(`novels/${book}_level${level}`);
        multiBar.log(chalk.blue(`Level ${level} summaries indexed and saved.\n`));

        // Prepare for next iteration
        currentSummaries = newSummaries;
        level++;
    } catch(error) {
        multiBar.log(chalk.red(`Error during Level ${level} summarization or FaissStore indexing: ${error.message}`));
        break; // Exit loop on error
    }
}
