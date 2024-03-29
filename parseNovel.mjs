import { OllamaEmbeddings } from '@langchain/community/embeddings/ollama';
import { FaissStore } from '@langchain/community/vectorstores/faiss';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { inspect } from 'node:util';
import cliProgress from 'cli-progress';

const book = 'Fighters_pages';

const loader = new PDFLoader(`novels/${book}.pdf`, { splitPages: false, });
const docs = await loader.load();

const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 4096,
    chunkOverlap: 128,
});

const splits = await splitter.splitDocuments(docs);

const embeddings = new OllamaEmbeddings({ model: 'nomic-embed-text', numCtx: 2048 });

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
