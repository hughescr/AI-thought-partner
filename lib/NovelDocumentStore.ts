import { Document } from '@langchain/core/documents';
import PouchDB from 'pouchdb';
import find from 'pouchdb-find';
PouchDB.plugin(find);
import { MarkdownChapterTextSplitter } from './MarkdownChapterTextSplitter';
import { ChapterDocumentStore } from './ChapterDocumentStore';
import { ChapterDocument } from './ChapterDocumentStore';
import type { ChapterSummaryGenerator } from './ChapterSummaryGenerator';
import _ from 'lodash';

import { FaissStore } from '@langchain/community/vectorstores/faiss';
import { NovelFaissStore } from './NovelFaissStore';
import type { Embeddings } from '@langchain/core/embeddings';

import type { MultiBar, SingleBar } from 'cli-progress';
import { rm } from 'node:fs/promises';

const NOVEL_DOCTYPE = 'novel';

// NovelDocument: represents the complete novel in Markdown.
export class NovelDocument extends Document<{
    docType: string
    novelID: string
    title: string
    author: string
    genre?: string
    filepath?: string
}> {
    constructor(fields: {
        pageContent: string
        metadata: { title: string, author: string, genre?: string, filepath?: string }
    }) {
        super({
            ...fields,
            metadata: {
                ...fields.metadata,
                docType: NOVEL_DOCTYPE,
                novelID: computeNovelID(fields.metadata.author, fields.metadata.title),
            },
        });
    }
}

// Helper to compute novelID reproducibly from author and title.
export function computeNovelID(author: string, title: string): string {
    // Simple reproducible ID, e.g. lowercased with spaces replaced by underscores.
    return `${_.chain(author).trim().toLower().replace(/\s+/g, '_').value()}_${_.chain(title).trim().toLower().replace(/\s+/g, '_').value()}`;
}

// NovelDocumentStore: stores novels and auto-splits them into chapters.
export class NovelDocumentStore {
    private db!: PouchDB.Database;
    private indexCreated: Promise<void>;
    private debugBar?: MultiBar;
    private chapterStore: ChapterDocumentStore;
    private chapterSplitter: MarkdownChapterTextSplitter;
    private filePath: string;
    public embeddings: Embeddings;
    private faissFolder: string;

    constructor(embeddings: Embeddings, dbConfig: { filePath: string, summaryGenerator: ChapterSummaryGenerator, debugBar?: MultiBar }) {
        this.debugBar = dbConfig.debugBar;
        this.db = new PouchDB(dbConfig.filePath, { auto_compaction: true });
        this.indexCreated = this.db
            .createIndex({ index: { fields: ['metadata.docType', 'metadata.novelID'] } })
            .then(_.noop);
        // Create our own ChapterDocumentStore using the same Loki instance.
        this.filePath = `${dbConfig.filePath}-FAISS`;
        this.embeddings = embeddings;
        this.chapterStore = new ChapterDocumentStore({ db: this.db, summaryGenerator: dbConfig.summaryGenerator, debugBar: this.debugBar });
        this.chapterSplitter = new MarkdownChapterTextSplitter();
        this.faissFolder = `${dbConfig.filePath}-FAISS`;
        // (No FAISS store creation here; we will do that per novel in addNovel.)
    }

    async addNovel(novelDoc: NovelDocument): Promise<void> {
        await this.indexCreated;
        // Persist the novel document.
        const storeNovel = novelDoc as NovelDocument & { _id: string };
        storeNovel._id = `novel_${novelDoc.metadata.novelID}`;
        await this.db.put(storeNovel);

        // Split the novel into chapters.
        const chapters = await this.chapterSplitter.splitDocuments([novelDoc]);
        const chapterBar = this.debugBar?.create(chapters.length, 0, { msg: 'Chapters' });
        const novelID = novelDoc.metadata.novelID;
        const storePath = `${this.faissFolder}/${novelID}`;
        await rm(storePath, { recursive: true, force: true }); // Remove any existing FAISS store
        const faiss = new FaissStore(this.embeddings, {});

        // Process each chapter using a helper.
        await Promise.all(_.map(chapters, (chapter, index) => this.processChapter(chapter, index + 1, novelDoc.metadata.novelID, faiss, chapterBar)));

        chapterBar?.stop();
        if(chapterBar) {
            this.debugBar?.remove(chapterBar);
        }
    }

    private async processChapter(
        chapter: Document,
        index: number,
        novelID: string,
        faiss: FaissStore,
        chapterBar: SingleBar | undefined
    ): Promise<void> {
        const chapterTitle = this.extractChapterTitle(chapter.pageContent);
        chapterBar?.update(index, { msg: chapterTitle });

        const chapterDoc = new ChapterDocument({
            pageContent: chapter.pageContent,
            metadata: { novelID, chapter: index },
        });
        await this.chapterStore.addChapter(chapterDoc);

        await this.processChapterSummary(chapterDoc, chapterTitle, faiss, chapterBar);
        await this.processChapterChunks(chapterDoc, chapterTitle, faiss, chapterBar);

        if(faiss._index) {
            await faiss.save(`${this.faissFolder}/${novelID}`);
        }
    }

    private extractChapterTitle(pageContent: string): string {
        return _.chain(pageContent).split('\n').head().trim().value();
    }

    private async processChapterSummary(
        chapter: ChapterDocument,
        chapterTitle: string,
        faiss: FaissStore,
        chapterBar: SingleBar | undefined
    ): Promise<void> {
        const summaryDoc = await this.chapterStore.getChapterSummary(chapter);
        if(summaryDoc) {
            chapterBar?.update(chapter.metadata.chapter, { msg: `Adding summary of ${chapterTitle} to FAISS` });
            await faiss.addDocuments([summaryDoc]);
        }
    }

    private async processChapterChunks(
        chapter: ChapterDocument,
        chapterTitle: string,
        faiss: FaissStore,
        chapterBar: SingleBar | undefined
    ): Promise<void> {
        chapterBar?.update(chapter.metadata.chapter, { msg: `Getting chunks of ${chapterTitle}` });
        const chunks = await this.chapterStore.getChapterChunks(chapter);
        chapterBar?.update(chapter.metadata.chapter, { msg: `Got ${chunks.length} chunks of ${chapterTitle}` });
        if(chunks.length > 0) {
            const chunkBar = this.debugBar?.create(chunks.length, 0, { msg: `Adding chunks of ${chapterTitle} to FAISS` });
            for(const chunk of chunks) {
                chunkBar?.increment();
                await faiss.addDocuments([chunk]);
            }
            chunkBar?.stop();
            if(chunkBar) {
                this.debugBar?.remove(chunkBar);
            }
        }
        chapterBar?.update(chapter.metadata.chapter, { msg: `Done with chunks of ${chapterTitle}` });
    }

    public async getVectorStoreForNovel(novel: NovelDocument): Promise<NovelFaissStore> {
        const storePath = `${this.faissFolder}/${novel.metadata.novelID}`;
        return await NovelFaissStore.load(storePath, this.embeddings, novel, this.chapterStore);
    }

    public async destroy(): Promise<void> {
        await this.db.destroy();
    }
}
