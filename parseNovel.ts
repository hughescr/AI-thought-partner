import {
    cachedJinaV2BaseENEmbeddings as embeddings,
    cachedJinaV2SmallENEmbeddings as fastEmbeddings,
    qwen25_14bLLM as summarizerLLM
} from './lib/LLMs.ts';
import { ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { FaissStore } from '@langchain/community/vectorstores/faiss';
import { TextLoader } from 'langchain/document_loaders/fs/text';
// import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import cliProgress from 'cli-progress';
import { SemanticTextSplitter } from './lib/SemanticTextSplitter.ts';
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
for(const chapter of chapterChunks) {
    chapterBar.increment();
    const smallerChunks = await splitter.splitDocuments([chapter]);

    const totalSmallerChunks = smallerChunks.length;
    // Create progress bar for smaller chunks
    const chunkBar: cliProgress.SingleBar = multiBar.create(totalSmallerChunks, 0, {
        name: 'Chunks',
    });

    for(const smallerChunk of smallerChunks) {
        const context = await contextChain.invoke({
            'long': chapter.pageContent,
            'short': smallerChunk.pageContent,
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

const splitChunks = _(splits)
    .flatten()
    .chunk(16)
    .value();
const bar: cliProgress.SingleBar = multiBar.create(
    _.flatten(splits).length,
    0,
    { name: 'Saving chunks' }
);
for(const chunk of splitChunks) {
    if(vectorStore) {
        await vectorStore.addDocuments(chunk);
    } else {
        vectorStore = await FaissStore.fromDocuments(chunk, embeddings); // Index with the slower, better embeddings
    }
    bar.increment(chunk.length);
}
if(vectorStore) {
    vectorStore.save(`novels/${book}`);
}
bar.stop();
multiBar.stop();
