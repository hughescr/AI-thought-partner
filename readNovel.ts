import { Command } from 'commander';
import _ from 'lodash';
import { cachedSnowflakeArctic2Embeddings as embeddings } from './lib/LLMs';
import { NovelDocumentStore, NovelDocument } from './lib/NovelDocumentStore';
import { logger } from '@hughescr/logger';
import { encoding_for_model } from '@dqbd/tiktoken';

function getTokenCount(text: string): number {
    const enc = encoding_for_model('o3-mini');
    const tokens = enc.encode(text);
    const tokenCount = tokens.length;
    enc.free();
    return tokenCount;
}

async function processDefault(store: NovelDocumentStore, novel: NovelDocument) {
    const novelSummary = await store.getNovelSummary(novel.metadata.novelID);
    if(novelSummary) {
        logger.info(`Novel Summary for "${novel.metadata.title}" by "${novel.metadata.author}":\n`);
        logger.info(novelSummary.pageContent);
        const summaryWordCount = _.words(novelSummary.pageContent).length;
        const summaryTokenCount = getTokenCount(novelSummary.pageContent);
        logger.info(`Summary Word Count: ${summaryWordCount}, Summary Token Count: ${summaryTokenCount}`);
    } else {
        logger.info(`No summary available for "${novel.metadata.title}" by "${novel.metadata.author}".`);
    }
    const fullWordCount = _.words(novel.pageContent).length;
    const fullTokenCount = getTokenCount(novel.pageContent);
    logger.info(`Full Novel Word Count: ${fullWordCount}, Full Novel Token Count: ${fullTokenCount}`);
}

async function processQuery(store: NovelDocumentStore, novel: NovelDocument, query: string, maxChapters: number) {
    const vectorStore = await store.getVectorStoreForNovel(novel);
    const retriever = vectorStore.asRetriever({ k: maxChapters });
    logger.info(`Searching for chapters matching query: "${query}"\n`);
    const retrievedChapters = await retriever.invoke(query);
    if(!retrievedChapters || retrievedChapters.length === 0) {
        logger.info('No relevant chapters found.');
        return;
    }
    for(const doc of retrievedChapters) {
        const chapterNum = doc.metadata.chapter;
        const summaryPreview = _.trim(doc.pageContent.slice(0, 200));
        const wordCount = _.words(doc.pageContent).length;
        const tokenCount = getTokenCount(doc.pageContent);
        logger.info(`Chapter ${chapterNum}:\nSummary Preview: ${summaryPreview}\nWord Count: ${wordCount}, Token Count: ${tokenCount}\n`);
    }
}

async function processInfo(store: NovelDocumentStore, novel: NovelDocument, title: string, author: string) {
    // Use the NovelDocumentStore API to retrieve chapter documents.
    const chapters = await store.getChapters(title, author);
    if(!chapters || chapters.length === 0) {
        logger.info('No chapters found for this novel.');
        return;
    }
    const chapterList = _(chapters).map((doc: { pageContent: string, metadata: { chapter: number } }) => {
        const firstLine = _.trim(_.head(_.split(doc.pageContent, '\n')) || '');
        const chapterTitle = _.replace(firstLine, /^#+\s*/, '');
        const wordCount = _.words(doc.pageContent).length;
        const tokenCount = getTokenCount(doc.pageContent);
        return { Chapter: doc.metadata.chapter, Title: chapterTitle, WordCount: wordCount, TokenCount: tokenCount };
    })
    .sortBy('Chapter')
    .value();

    logger.info(`Novel Info:
Title: ${novel.metadata.title}
Author: ${novel.metadata.author}
Genre: ${novel.metadata.genre || 'N/A'}
Number of Chapters: ${chapters.length}\n`);
    logger.info(JSON.stringify(chapterList, null, 2));
}

async function main() {
    const program = new Command();
    program
    .name('readNovel')
    .description('Read and query a novel from the store')
    .argument('<title>', 'Novel title')
    .argument('<author>', 'Novel author')
    .option('--query <queryStr>', 'Query to lookup relevant chapters')
    .option('--maxChapters <number>', 'Maximum number of chapters to retrieve (default: 3)', '3')
    .option('--info', 'Show detailed novel info including chapter listing')
    .parse(process.argv);

    const options = program.opts();
    const [title, author] = program.args;

    // Validate mutually exclusive options: --query and --info cannot be used together.
    if(options.query && options.info) {
        throw new Error('Error: Options --query and --info are mutually exclusive.');
    }

    // Instantiate NovelDocumentStore with a default filePath.
    const store = new NovelDocumentStore(embeddings, { filePath: 'novels_db' });

    // Look up the novel.
    const novel = await store.getNovel(title, author);
    if(!novel) {
        throw new Error(`Novel "${title}" by "${author}" not found in the store.`);
    }

    if(!options.query && !options.info) {
        await processDefault(store, novel);
    } else if(options.query) {
        const maxChapters = parseInt(options.maxChapters, 10) || 3;
        await processQuery(store, novel, options.query, maxChapters);
    } else if(options.info) {
        await processInfo(store, novel, title, author);
    }
}

main().catch((err) => {
    logger.error('An error occurred:', err);
    throw err;
});
