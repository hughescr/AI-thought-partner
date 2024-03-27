import { OllamaEmbeddings } from '@langchain/community/embeddings/ollama';
import { FaissStore } from '@langchain/community/vectorstores/faiss';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { inspect } from 'node:util';
import cliProgress from 'cli-progress';

const book = 'Exile draft 2';

const loader = new TextLoader(`novels/${book}.txt`);
const docs = await loader.load();

const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 2048,
    chunkOverlap: 128,
});

const splits = await splitter.splitDocuments(docs);

const embeddings = new OllamaEmbeddings({ model: 'nomic-embed-text', requestOptions: { numGpu: 1 }});

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
