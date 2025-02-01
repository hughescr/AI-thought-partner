import {
    novaLiteLLM as summarizerLLM,
    cachedSnowflakeArctic2Embeddings
} from './lib/LLMs';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import cliProgress from 'cli-progress';
import { FaissStore } from '@langchain/community/vectorstores/faiss';
import { SemanticTextSplitter } from './lib/SemanticTextSplitter';
import { MarkdownChapterTextSplitter } from './lib/MarkdownChapterTextSplitter';
import { Document } from '@langchain/core/documents';
import { ChapterSummaryGenerator } from './lib/ChapterSummaryGenerator';
import _ from 'lodash';
import { logger } from '@hughescr/logger';
import pThrottle from 'p-throttle';
import pLimit from 'p-limit';
import fs from 'fs/promises';

/**
 * Saves an array of summaries to a plain text file.
 * @param summaries - Array of Document objects containing summaries.
 * @param book - The name of the book being processed.
 */
async function saveSummariesToFile(chapters: Document[], book: string): Promise<void> {
    const filePath = `novels/${book}_chapter_summaries.txt`;
    const content = _(chapters).map('metadata.summary').join('\n\n');
    await fs.writeFile(filePath, content, 'utf-8');
    logger.info(`Summaries saved to ${filePath}\n`);
}

_.mixin({
    awaitAll: function <T>(promiseArray: Promise<T>[]) {
        return Promise.all(promiseArray);
    },
},
{ chain: true } // Enable chaining for this mixin
);

const idealContextSize = 8192; // Desired total token limit for collection of 10 summaries (from future retrieval)
const book = 'Christmas Town query version';
// const loader = new PDFLoader(`novels/${book}.pdf`, { splitPages: true });
const loader = new TextLoader(`novels/${book}.md`);
const docs = await loader.load();

// Strategy per https://blog.getbind.co/2024/09/25/claude-contextual-retrieval-vs-rag-how-is-it-different/
// Break the source into large chunks (maybe 8k tokens semantically)
// Then, for each large chunk, break it into small chunks (maybe 256 tokens), and ask an LLM to describe the context of each small chunk (adding another 256 tokens)
// Then, concat the context and the extract, calculate encodings, and store in a vector store

const chapterSplitter = new MarkdownChapterTextSplitter();

const targetSummarySize = idealContextSize / 10;

const summaryGenerator = new ChapterSummaryGenerator({
    llm: summarizerLLM,
    targetSummarySize: targetSummarySize, // 400 tokens
});

const limit = pLimit(32); // Limit to 32 concurrent requests
const throttle = pThrottle({
    limit: 64,
    interval: 60 * 1000,
}); // Limit to 64 per minute

const throttledSummaryGenerator = throttle(({ doc }: { doc: Document }) => summaryGenerator.generateSummary(doc));
const limitedThrottledSummaryGenerator = ({ doc }: { doc: Document }) => limit(() => throttledSummaryGenerator({ doc }));

// Initialize variables for iterative summarization
const chapters = await chapterSplitter.splitDocuments(docs);

let bar = new cliProgress.SingleBar({
    format: 'Processing [{bar}] {percentage}% | ETA: {eta}s | {value}/{total} chunks',
    barCompleteChar: '\u2588',
    barIncompleteChar: '\u2591',
    hideCursor: false,
});
bar.start(chapters.length, 0);

if(_.includes(summarizerLLM.lc_namespace, 'ollama')) {
    // Serial Processing for Ollama
    for(const chapter of chapters) {
        if(!chapter.metadata) {
            chapter.metadata = {};
        }
        chapter.metadata.summary = (await summaryGenerator.generateSummary(chapter)).pageContent;
        bar.increment();
    }
} else {
    // Parallel Processing for Non-Ollama LLMs using limitedThrottledSummaryGenerator
    const batchPromises = _.map(chapters, chapter =>
        limitedThrottledSummaryGenerator({ doc: chapter })
                .then((summary) => {
                    bar.increment();
                    return summary;
                })
                .catch((error) => {
                    logger.error(`Error generating summary for batch: ${error.message}`);
                    bar.increment();
                    return null;
                })
    );

    const summaries = await Promise.all(batchPromises);
    _.forEach(chapters, (chapter, i) => {
        if(!chapter.metadata) {
            chapter.metadata = {};
        }
        chapter.metadata.summary = summaries[i]?.pageContent;
    });
}
bar.stop();

const embeddingsArray = [];
// For each chapter, create embeddings and prepare for FAISS indexing
for(const chapter of chapters) {
    const summaryEmbedding = await cachedSnowflakeArctic2Embeddings.embedQuery(chapter.metadata.summary);
    embeddingsArray.push(summaryEmbedding);
}

const store = new FaissStore(cachedSnowflakeArctic2Embeddings, {});
const docIDs = await store.addVectors(embeddingsArray, chapters);

const contextLength = 8192; // Context length of the embedding model
const chunkSize = Math.floor((3 / 4) * contextLength); // 6144 tokens

const semanticSplitter = new SemanticTextSplitter({
    embeddings: cachedSnowflakeArctic2Embeddings,
    chunkSize: chunkSize,
    embeddingBatchSize: 16, // Adjust as needed
    showProgress: false,    // Set to true to display progress
});

bar = new cliProgress.SingleBar({
    format: 'Splitting [{bar}] {percentage}% | ETA: {eta}s | {value}/{total} chapters',
    barCompleteChar: '\u2588',
    barIncompleteChar: '\u2591',
    hideCursor: false,
});
bar.start(chapters.length, 0);
_.forEach(chapters, async (chapter, i) => {
    // Semantically split the chapter
    const chunks = await semanticSplitter.splitText(chapter.pageContent);

    // Calculate embeddings for each semantic chunk
    const chunkEmbeddings = await cachedSnowflakeArctic2Embeddings.embedDocuments(chunks);

    const repeatChapter = _.times(chunks.length, () => chapter);
    const repeatChapterIDs = _.times(chunks.length, () => docIDs[i]);

    // Now add these to the store
    await store.addVectors(chunkEmbeddings, repeatChapter, { ids: repeatChapterIDs });
    bar.increment();
});
bar.stop();

// Save the FAISS store to disk
const faissDirectory = `novels/${book}_faiss`;
await store.save(faissDirectory);
await saveSummariesToFile(chapters, book);
