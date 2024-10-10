// TODO: Add BM25 for search

import { ChatOllama, OllamaEmbeddings } from '@langchain/ollama';
import { ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { CacheBackedEmbeddings } from 'langchain/embeddings/cache_backed';
import { InMemoryStore } from 'langchain/storage/in_memory';
import { FaissStore } from '@langchain/community/vectorstores/faiss';
import { TextLoader } from 'langchain/document_loaders/fs/text';
// import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import cliProgress from 'cli-progress';
import { SemanticTextSplitter } from './lib/SemanticTextSplitter';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import _ from 'lodash';

_.mixin({
    awaitAll: function (promiseArray) {
        return Promise.all(promiseArray);
    }
}, { chain: true }); // Enable chaining for this mixin

const book = 'Christmas Town beta';
// const loader = new PDFLoader(`novels/${book}.pdf`, { splitPages: true, });
const loader = new TextLoader(`novels/${book}.md`);
const docs = await loader.load();

// Strategy per https://blog.getbind.co/2024/09/25/claude-contextual-retrieval-vs-rag-how-is-it-different/
// Break the source into large chunks (maybe 8k tokens semantically)
// Then, for each large chunk, break it into small chunks (maybe 256 tokens), and ask an LLM to describe the context of each small chunk (adding another 256 tokens)
// Then, concat the context and the extract, calculate encodings, and store in a vector store

const fastEmbeddings = new OllamaEmbeddings({ model: 'nomic-embed-text', requestOptions: { numCtx: 2048 } });
// const coreEmbeddings = new OllamaEmbeddings({ model: 'mxbai-embed-large', requestOptions: { numCtx: 512 } });
// const coreEmbeddings = new OllamaEmbeddings({ model: 'bge-large', requestOptions: { numCtx: 512 } });
const coreEmbeddings = new OllamaEmbeddings({ model: 'bge-m3', requestOptions: { numCtx: 8192 } });
// const coreEmbeddings = new OllamaEmbeddings({ model: 'mistral:7b-instruct-v0.2-q8_0', requestOptions: { numCtx: 32768 } });
// const coreEmbeddings = new OllamaEmbeddings({ model: 'llama3.1:8b-instruct-q8_0', requestOptions: { numCtx: 2048 } });

const commonOptions = { temperature: 1, seed: 19740822, keepAlive: '15m' };
const commonOptions32k = { numCtx: 32 * 1024, ...commonOptions };
// Prompt parse: ~500-1500 t/s; generation: ~60-70 t/s; HAS TOOLS
const qwen25_3bLLM = new ChatOllama({ model: 'qwen2.5:3b-instruct-q8_0', ...commonOptions32k });

const chapterSplitter = new RecursiveCharacterTextSplitter({
    separators: ['\n#', '\n\n', '.', '!', '?'], // Chapters, paragraphs, sentences
    chunkSize: 20 * 1024,
    keepSeparators: true,
    chunkOverlap: 0,
});
chapterSplitter.splitOnSeparator = function(text, separator) {
    let splits;
    if(separator) {
        if(this.keepSeparator) {
            const regexEscapedSeparator = _.replace(separator, /[/\-\\^$*+?.()|[\]{}]/g, '\\$&');
            splits = _.split(text, new RegExp(`(?<=${regexEscapedSeparator})`));
        } else {
            splits = _.split(text, separator);
        }
    } else {
        splits = _.split(text, '');
    }
    return _.filter(splits, s => s !== '');
};
chapterSplitter.splitOnSeparator = chapterSplitter.splitOnSeparator.bind(chapterSplitter);

const chapterChunks = await chapterSplitter.splitDocuments(docs);

const embeddings = CacheBackedEmbeddings.fromBytesStore(
    coreEmbeddings,
    new InMemoryStore(),
    {
        namespace: coreEmbeddings.modelName,
    }
);

const splitter = new SemanticTextSplitter({
    showProgress: false,
    initialChunkSize: 32, // Tokens!
    chunkSize: 512, // Tokens!
    embeddings: fastEmbeddings, // Use fast embeddings for decent semantic splits
    embeddingBatchSize: 128,
});

// Now go through each chapter, split it into smaller chunks, and then calculate context for each chunk.
const contextSummaryPrompt = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate('Given a long passage and a short passage, generate a short explanatory context that grounds the short passage. Output just the context with no JSON or other markup.'),
    HumanMessagePromptTemplate.fromTemplate('{{ "long": {long}, "short": {short}}}'),
]);
const contextChain = contextSummaryPrompt.pipe(qwen25_3bLLM).pipe(new StringOutputParser());

// Initialize MultiBar
const multiBar = new cliProgress.MultiBar({
    clearOnComplete: false,
    hideCursor: true,
    format: '{bar} {value}/{total} {name} | {percentage}% | Time: {duration_formatted} | ETA: {eta_formatted}'
}, cliProgress.Presets.shades_classic);
const totalChapters = chapterChunks.length;
// Create main progress bar for chapters
const chapterBar = multiBar.create(totalChapters, 0, { name: 'Chapters' });

const splits = [];
for(const chapter of chapterChunks) {
    chapterBar.increment();
    const smallerChunks = await splitter.splitDocuments([chapter]);

    const totalSmallerChunks = smallerChunks.length;
    // Create progress bar for smaller chunks
    const chunkBar = multiBar.create(totalSmallerChunks, 0, { name: 'Chunks' });

    for(const smallerChunk of smallerChunks) {
        const context = await contextChain.invoke({
            'long': JSON.stringify(chapter.pageContent),
            'short': JSON.stringify(smallerChunk.pageContent),
        });
        chunkBar.increment();
        splits.push({
            ...smallerChunk,
            context,
        });
    }

    // Stop the smaller chunks progress bar
    chunkBar.stop();
    multiBar.remove(chunkBar);
}
// Stop the chapter progress bar and MultiBar
chapterBar.stop();

let vectorStore;

const splitChunks = _(splits).flatten().chunk(16).value();
const bar = multiBar.create(_.flatten(splits).length, 0, { name: 'Saving chunks' });
for(const chunk of splitChunks) {
    if(vectorStore) {
        await vectorStore.addDocuments(chunk);
    } else {
        vectorStore = await FaissStore.fromDocuments(chunk, embeddings); // Index with the slower, better embeddings
    }
    bar.increment(chunk.length);
}
vectorStore.save(`novels/${book}`);
bar.stop();
multiBar.stop();
