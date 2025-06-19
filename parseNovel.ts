import {
    qwen3_30bA3bLLM as summarizerLLM,
    cachedSnowflakeArctic2Embeddings as embeddings
} from './lib/LLMs';

import { NovelDocument, NovelDocumentStore } from './lib/NovelDocumentStore';
import { ChapterSummaryGenerator } from './lib/ChapterSummaryGenerator';

// import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
// import { DocxLoader } from '@langchain/community/document_loaders/fs/docx';
import { TextLoader } from 'langchain/document_loaders/fs/text';

import _ from 'lodash';
import { MultiBar, Presets as cliProgressPresets } from 'cli-progress';
import { Command } from 'commander';
import { logger } from '@hughescr/logger';
import chalk from 'chalk';

if(process.versions.bun === undefined) {
    logger.info(chalk.greenBright('Running under Node, setting global dispatcher'));
    const { setGlobalDispatcher, Agent } = await import('undici');
    setGlobalDispatcher(new Agent({ headersTimeout: 0, bodyTimeout: 0 })); // ensure we wait for long ollama runs
} else {
    logger.warn(chalk.yellowBright('Running under Bun, not setting global dispatcher so LLMs might timeout'));
}

const program = new Command();
program
    .description('Parse a novel file and add it to the database')
    .requiredOption('-t, --title <title>', 'title of the novel')
    .requiredOption('-a, --author <author>', 'author of the novel')
    .option('-g, --genre <genre>', 'genre of the novel', 'Unknown')
    .option('-f, --format <format>', 'file format (md, pdf, docx)', 'md')
    .parse(process.argv);

const options = program.opts();

const book = options.title;
const filepath = `novels/${book}.${options.format}`;

// Select loader based on file format
let loader;
if(options.format === 'md') {
    loader = new TextLoader(filepath);
} else {
    throw new Error(`Unsupported format: ${options.format}. Please use md.`);
}

const docs = await loader.load();
const novel = NovelDocument.fromDocument(docs[0], book, options.author, options.genre, loader.filePathOrBlob.toString());

const summaryGenerator = new ChapterSummaryGenerator({
    llm: summarizerLLM,
    targetSummarySize: 256,
});

const bars = new MultiBar({
    clearOnComplete: true,
    hideCursor: false,
    format: '{bar} {percentage}% | {duration_formatted} | ETA: {eta_formatted} | {value}/{total} | {msg}',
}, cliProgressPresets.shades_classic);
const novelStore = new NovelDocumentStore(embeddings, { filePath: 'novels_db', summaryGenerator: summaryGenerator, debugBar: bars });
if(!(await novelStore.getNovel(novel.metadata.title, novel.metadata.author))) {
    await novelStore.addNovel(novel);
}
bars.stop();

logger.info('Novel:', novel.metadata);
const novelSummary = await novelStore.getNovelSummary(novel.metadata.novelID);
logger.info(`Novel Summary: ${novelSummary?.pageContent}`);
