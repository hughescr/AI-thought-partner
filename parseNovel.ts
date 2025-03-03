import {
    deepseekR1_Qwen32bLLM as summarizerLLM,
    cachedSnowflakeArctic2Embeddings
} from './lib/LLMs';

import { NovelDocument, NovelDocumentStore } from './lib/NovelDocumentStore';
import { ChapterSummaryGenerator } from './lib/ChapterSummaryGenerator';

// import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
// import { DocxLoader } from '@langchain/community/document_loaders/fs/docx';
import { TextLoader } from 'langchain/document_loaders/fs/text';

import _ from 'lodash';
import { MultiBar, Presets as cliProgressPresets } from 'cli-progress';

const book = 'Christmas Town query version';
// const loader = new PDFLoader(`novels/${book}.pdf`, { splitPages: true });
// const loader = new DocxLoader(`novels/${book}.docx`);
const loader = new TextLoader(`novels/${book}.md`);
const docs = await loader.load();
const novel = NovelDocument.fromDocument(docs[0], book, 'Erica S. Hughes', 'Young Adult', loader.filePathOrBlob.toString());

const summaryGenerator = new ChapterSummaryGenerator({
    llm: summarizerLLM,
    targetSummarySize: 256,
});

const bars = new MultiBar({
    clearOnComplete: true,
    hideCursor: false,
    format: '{bar} {percentage}% | {duration_formatted} | ETA: {eta_formatted} | {value}/{total} | {msg}',
}, cliProgressPresets.shades_classic);
const novelStore = new NovelDocumentStore(cachedSnowflakeArctic2Embeddings, { filePath: 'novels_db', summaryGenerator: summaryGenerator, debugBar: bars });
if(!(await novelStore.getNovel(novel.metadata.title, novel.metadata.author))) {
    await novelStore.addNovel(novel);
}
bars.stop();

console.log('Novel:', novel.metadata);
const novelSummary = await novelStore.getNovelSummary(novel.metadata.novelID);
console.log('Novel Summary:', novelSummary?.pageContent);
