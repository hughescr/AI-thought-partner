import { OllamaEmbeddings } from '@langchain/ollama';
import { CacheBackedEmbeddings } from "langchain/embeddings/cache_backed";
import { InMemoryStore } from "langchain/storage/in_memory";
import { FaissStore } from '@langchain/community/vectorstores/faiss';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import cliProgress from 'cli-progress';
import { SemanticTextSplitter } from './lib/SemanticChunker.mjs';
import _ from 'lodash';
import { logger } from '@hughescr/logger';

const book = 'Christmas Town beta';
// const loader = new PDFLoader(`novels/${book}.pdf`, { splitPages: true, });
const loader = new TextLoader(`novels/${book}.md`);
const docs = await loader.load();

// const docs = [new Document({ pageContent: 'It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of light, it was the season of darkness, it was the spring of hope, it was the winter of despair.' })];

// const coreEmbeddings = new OllamaEmbeddings({ model: 'nomic-embed-text', requestOptions: { numCtx: 2048 } });
// const coreEmbeddings = new OllamaEmbeddings({ model: 'mxbai-embed-large', requestOptions: { numCtx: 512 } });
// const coreEmbeddings = new OllamaEmbeddings({ model: 'bge-large', requestOptions: { numCtx: 512 } });
const coreEmbeddings = new OllamaEmbeddings({ model: 'bge-m3', requestOptions: { numCtx: 8192 } });
// const coreEmbeddings = new OllamaEmbeddings({ model: 'mistral:7b-instruct-v0.2-q8_0', requestOptions: { numCtx: 32768 } });
// const coreEmbeddings = new OllamaEmbeddings({ model: 'llama3.1:8b-instruct-q8_0', requestOptions: { numCtx: 2048 } });

const embeddings = CacheBackedEmbeddings.fromBytesStore(
    coreEmbeddings,
    new InMemoryStore(),
    {
        namespace: coreEmbeddings.modelName,
    }
)

const splitter = new SemanticTextSplitter({
    showProgress: true,
    initialChunkSize: 128, // Tokens!
    chunkSize: 512, // Tokens!
    embeddings,
});

const splits = await splitter.splitDocuments(docs);
// logger.info(splits);

let vectorStore;

const bar = new cliProgress.SingleBar({ barsize: 80, format: '{bar} {value}/{total} splits | {percentage}% | Time: {duration_formatted} | ETA: {eta_formatted}'}, cliProgress.Presets.shades_classic);
bar.start(_.flatten(splits).length, 0);
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
