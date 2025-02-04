import {
    novaLiteLLM as summarizerLLM,
    cachedSnowflakeArctic2Embeddings
} from './lib/LLMs';

import { NovelDocument, NovelDocumentStore } from './lib/NovelDocumentStore';
import { ChapterSummaryGenerator } from './lib/ChapterSummaryGenerator';

import { TextLoader } from 'langchain/document_loaders/fs/text';

import _ from 'lodash';
import { MultiBar, Presets as cliProgressPresets } from 'cli-progress';

const book = 'Christmas Town query version';
// const loader = new PDFLoader(`novels/${book}.pdf`, { splitPages: true });
const loader = new TextLoader(`novels/${book}.md`);
const docs = await loader.load();
const novel = docs[0] as NovelDocument;
novel.metadata.title = book;
novel.metadata.author = 'Erica S. Hughes';
novel.metadata.genre = 'Young adult';

const summaryGenerator = new ChapterSummaryGenerator({
    llm: summarizerLLM,
    targetSummarySize: 256,
});
const bars = new MultiBar({
    clearOnComplete: true,
    hideCursor: false,
    format: '{bar} {percentage}% | {duration_formatted} | ETA: {eta_formatted} | {value}/{total} | {msg}',
}, cliProgressPresets.shades_classic);
const novelStore = new NovelDocumentStore(cachedSnowflakeArctic2Embeddings, { filePath: 'novels.db', summaryGenerator: summaryGenerator, debugBar: bars });
await novelStore.addNovel(novel);

await novelStore.close();

bars.stop();
