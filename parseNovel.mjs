import { OllamaEmbeddings } from '@langchain/community/embeddings/ollama';
import { CacheBackedEmbeddings } from "langchain/embeddings/cache_backed";
import { InMemoryStore } from "langchain/storage/in_memory";
import { FaissStore } from '@langchain/community/vectorstores/faiss';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { Document } from "langchain/document";
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { inspect } from 'node:util';
import cliProgress from 'cli-progress';
import { SemanticTextSplitter } from './lib/SemanticChunker.mjs';
import _ from 'lodash';

const book = 'Christmas Town draft 2';
// const loader = new PDFLoader(`novels/${book}.pdf`, { splitPages: true, });
const loader = new TextLoader(`novels/${book}.txt`);
const docs = await loader.load();

// const docs = [new Document({ pageContent: 'It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of light, it was the season of darkness, it was the spring of hope, it was the winter of despair.' })];

const fastEmbeddings = new OllamaEmbeddings({ model: 'nomic-embed-text', numCtx: 2048 });
const midEmbeddings = new OllamaEmbeddings({ model: 'mxbai-embed-large', numCtx: 512 });
const slowEmbeddings = new OllamaEmbeddings({ model: 'mistral:7b-instruct-v0.2-q8_0', numCtx: 32768 });

const embeddings = CacheBackedEmbeddings.fromBytesStore(
    midEmbeddings,
    new InMemoryStore(),
    {
        namespace: midEmbeddings.modelName,
    }
)

const splitter = new SemanticTextSplitter({
    showProgress: true,
    chunkSize: 100, // Tokens!
    embeddings,
});

const splits = await splitter.splitDocuments(docs);
// console.log(inspect(splits, { depth: null }));

let vectorStore;

const bar = new cliProgress.SingleBar({ barsize: 80, format: '{bar} {value}/{total} splits | {percentage}% | Time: {duration_formatted} | ETA: {eta_formatted}'}, cliProgress.Presets.shades_classic);
bar.start(splits.length, 0);
for(let split of splits) {
    if(vectorStore) {
        await vectorStore.addDocuments([split]);
    } else {
        vectorStore = await FaissStore.fromDocuments([split], embeddings);
    }
    bar.increment();
}
vectorStore.save(`novels/${book}`);
bar.stop();
