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
import { Document } from '@langchain/core/documents';
import _ from 'lodash';

_.mixin({
        awaitAll: function <T>(promiseArray: Promise<T>[]) {
            return Promise.all(promiseArray);
        },
    },
    { chain: true } // Enable chaining for this mixin
);

const book = 'Christmas Town beta';
// const loader = new PDFLoader(`novels/${book}.pdf`, { splitPages: true });
const loader = new TextLoader(`novels/${book}.md`);
const docs = await loader.load();

// Strategy per https://blog.getbind.co/2024/09/25/claude-contextual-retrieval-vs-rag-how-is-it-different/
// Break the source into large chunks (maybe 8k tokens semantically)
// Then, for each large chunk, break it into small chunks (maybe 256 tokens), and ask an LLM to describe the context of each small chunk (adding another 256 tokens)
// Then, concat the context and the extract, calculate encodings, and store in a vector store

const fastEmbeddings = new OllamaEmbeddings({
    model: 'nomic-embed-text',
    requestOptions: { numCtx: 2048 },
});
// const coreEmbeddings: OllamaEmbeddings = new OllamaEmbeddings({ model: 'mxbai-embed-large', requestOptions: { numCtx: 512 } });
// const coreEmbeddings: OllamaEmbeddings = new OllamaEmbeddings({ model: 'bge-large', requestOptions: { numCtx: 512 } });
const coreEmbeddings = new OllamaEmbeddings({
    model: 'bge-m3',
    requestOptions: { numCtx: 8192 },
});
// const coreEmbeddings: OllamaEmbeddings = new OllamaEmbeddings({ model: 'mistral:7b-instruct-v0.2-q8_0', requestOptions: { numCtx: 32768 } });
// const coreEmbeddings: OllamaEmbeddings = new OllamaEmbeddings({ model: 'llama3.1:8b-instruct-q8_0', requestOptions: { numCtx: 2048 } });

interface CommonOptions {
    temperature: number;
    seed: number;
    keepAlive: string;
}

const commonOptions = {
    temperature: 1,
    seed: 19740822,
    keepAlive: '15m',
};
const commonOptions32k = {
    numCtx: 32 * 1024,
    ...commonOptions,
};

const summarizerLLM = new ChatOllama({
    ...commonOptions32k,
    // model: 'qwen2.5:32b-instruct-q8_0',
    // model: 'mistral-small:22b-instruct-2409-q8_0',
    // model: 'llama3.1:8b-instruct-q8_0',
    model: 'command-r:35b-08-2024-q8_0',
});

class RecursiveCharacterTextSplitterSeparatorMod extends RecursiveCharacterTextSplitter {
    // Override the splitOnSeparator method to allow for keeping the separator attached to the earlier chunk not the later chunk
    // Without this, punctuation ends up on the wrong chunk... for example
    // Sentence 1. Sentence 2. ==> ['Sentence 1', '. Sentence 2', '.'] instead of ['Sentence 1.', 'Sentence 2.' ]
    // The former is clearly dumber than shit and will confuse the LLM with its weird leading periods and no end to the sentence, etc.
    splitOnSeparator(text: string, separator: string): string[] {
        let splits: string[] = [];
        if (separator) {
            if (this.keepSeparator) {
                const regexEscapedSeparator: string = separator.replace(
                    /[/\-\\^$*+?.()|[\]{}]/g,
                    '\\$&'
                );
                splits = text.split(new RegExp(`(?<=${regexEscapedSeparator})`));
            } else {
                splits = text.split(separator);
            }
        } else {
            splits = text.split('');
        }
        return splits.filter((s) => s !== '');
    }
};

const chapterSplitter = new RecursiveCharacterTextSplitterSeparatorMod({
    separators: ['\n#', '\n\n', '.', '!', '?'], // Chapters, paragraphs, sentences
    chunkSize: 20 * 1024,
    keepSeparator: true,
    chunkOverlap: 0,
});

const chapterChunks = await chapterSplitter.splitDocuments(docs);

const embeddings: CacheBackedEmbeddings = CacheBackedEmbeddings.fromBytesStore(
    coreEmbeddings,
    new InMemoryStore(),
    {
        namespace: coreEmbeddings.model,
    }
);

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
        'Given a long passage and a short passage, generate a very minimal, short explanatory context that grounds the short passage, and clarifies who any pronouns refer to, if that is not clear in the short extract alone. Output just the generated context with no JSON nor other markup or lead-in, just the raw text by itself. Do not put quotation marks around the context or anything like that.'
    ),
    HumanMessagePromptTemplate.fromTemplate(`Long passage:
{long}

Short passage:
{short}`),
]);

const contextChain = contextSummaryPrompt
    .pipe(summarizerLLM)
    .pipe(new StringOutputParser());

// Initialize MultiBar
const multiBar: cliProgress.MultiBar = new cliProgress.MultiBar(
    {
        clearOnComplete: false,
        hideCursor: true,
        format:
            '{bar} {value}/{total} {name} | {percentage}% | Time: {duration_formatted} | ETA: {eta_formatted}',
    },
    cliProgress.Presets.shades_classic
);
const totalChapters: number = chapterChunks.length;
// Create main progress bar for chapters
const chapterBar: cliProgress.SingleBar = multiBar.create(totalChapters, 0, {
    name: 'Chapters',
});

const splits: Document[] = [];
for (const chapter of chapterChunks) {
    chapterBar.increment();
    const smallerChunks = await splitter.splitDocuments([chapter]);

    const totalSmallerChunks = smallerChunks.length;
    // Create progress bar for smaller chunks
    const chunkBar: cliProgress.SingleBar = multiBar.create(totalSmallerChunks, 0, {
        name: 'Chunks',
    });

    for (const smallerChunk of smallerChunks) {
        const context = await contextChain.invoke({
            long: chapter.pageContent,
            short: smallerChunk.pageContent,
        });
        multiBar.log(`Context: (${context.length}) ${context}\n`);
        chunkBar.increment();
        smallerChunk.metadata.context = context; // Save the context in the metadata
        splits.push(smallerChunk);
    }

    // Stop the smaller chunks progress bar
    chunkBar.stop();
    multiBar.remove(chunkBar);
}
// Stop the chapter progress bar and MultiBar
chapterBar.stop();

let vectorStore: FaissStore | undefined;

const splitChunks = _.chain(splits)
    .flatten()
    .chunk(16)
    .value();
const bar: cliProgress.SingleBar = multiBar.create(
    _.flatten(splits).length,
    0,
    { name: 'Saving chunks' }
);
for (const chunk of splitChunks) {
    if (vectorStore) {
        await vectorStore.addDocuments(chunk);
    } else {
        vectorStore = await FaissStore.fromDocuments(chunk, embeddings); // Index with the slower, better embeddings
    }
    bar.increment(chunk.length);
}
if (vectorStore) {
    vectorStore.save(`novels/${book}`);
}
bar.stop();
multiBar.stop();
