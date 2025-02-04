import { Document } from '@langchain/core/documents';
import Loki from 'lokijs';
import { promisify } from 'node:util';
import { ChapterSummaryDocumentStore, ChapterSummaryDocument } from './ChapterSummaryDocumentStore';
import { ChapterChunkDocumentStore, ChapterChunkDocument } from './ChapterChunkDocumentStore';
import { ChapterSummaryGenerator } from './ChapterSummaryGenerator';

import type { MultiBar } from 'cli-progress';
import _ from 'lodash';

export class ChapterDocument extends Document<{ novelID: string, chapter: number }> {
    constructor(fields: { pageContent: string, metadata: { novelID: string, chapter: number } }) {
        super(fields);
    }
}

export class ChapterDocumentStore {
    private db!: Loki;
    private collection!: Loki.Collection;
    private debugBar?: MultiBar;
    private summaryStore!: ChapterSummaryDocumentStore;
    private summaryGenerator: ChapterSummaryGenerator;
    private chapterChunkStore!: ChapterChunkDocumentStore;
    private loadPromise: Promise<void>;

    constructor(config: { db: Loki, summaryGenerator: ChapterSummaryGenerator, debugBar?: MultiBar }) {
        this.debugBar = config.debugBar;
        this.summaryGenerator = config.summaryGenerator;
        this.db = config.db;
        this.loadPromise = Promise.resolve();
        // Create (or get) the chapters collection with unique compound index on novelID and chapter.
        this.collection =
            this.db.getCollection('chapters') ||
            this.db.addCollection('chapters', {
                unique: ['metadata.novelID', 'metadata.chapter'],
                indices: ['metadata.novelID', 'metadata.chapter'],
                autoupdate: true,
            });
        // Pass the same db instance to the summary store.
        this.summaryStore = new ChapterSummaryDocumentStore({ db: this.db, debugBar: this.debugBar });
        this.chapterChunkStore = new ChapterChunkDocumentStore({ db: this.db, debugBar: this.debugBar });
    }

    private attachAutoUpdate(doc: ChapterDocument): ChapterDocument {
        let currentContent = doc.pageContent;
        Object.defineProperty(doc, 'pageContent', {
            get: () => currentContent,
            set: (newVal) => {
                currentContent = newVal;
                this.collection.update(doc);
                (async () => {
                    const summaryResult = await this.summaryGenerator.generateSummary(
                        new Document({ pageContent: newVal, metadata: doc.metadata })
                    );
                    const existing = await this.summaryStore.getChapterSummary(doc.metadata.chapter);
                    if(existing) {
                        existing.pageContent = summaryResult.pageContent;
                        // Remove explicit save call here.
                    } else {
                        summaryResult.metadata.chapter = doc.metadata.chapter;
                        await this.summaryStore.addChapterSummary(summaryResult as ChapterSummaryDocument);
                    }
                    await this.chapterChunkStore.deleteChapterChapters(doc.metadata.chapter);
                    await this.chapterChunkStore.addChunksForChapter(doc.metadata.chapter, newVal);
                })();
            },
            configurable: true
        });
        return doc;
    }

    private async generateAndStoreSummary(doc: ChapterDocument): Promise<void> {
        const chapterTitle = _.chain(doc.pageContent).split('\n').head()?.trim().trim('#').trim().value();
        const bar = this.debugBar?.create(1, 0, { msg: `Generating summary for ${chapterTitle}` });
        const summaryDoc = await this.summaryGenerator.generateSummary(doc);
        bar?.stop();
        if(bar) {
            this.debugBar?.remove(bar);
        }
        summaryDoc.metadata.chapter = doc.metadata.chapter;
        await this.summaryStore.addChapterSummary(summaryDoc as ChapterSummaryDocument);
    }

    async addChapter(doc: ChapterDocument): Promise<void> {
        if(!doc.metadata || !_.isNumber(doc.metadata.chapter) || !doc.metadata.novelID) {
            throw new Error('chapter metadata is required');
        }
        if(this.collection.findOne({ 'metadata.novelID': doc.metadata.novelID, 'metadata.chapter': doc.metadata.chapter })) {
            throw new Error('Document is already in collection, please use update()');
        }
        this.collection.insert(doc);
        // No explicit save here—autosave will handle it.
        await this.generateAndStoreSummary(doc);
        await this.chapterChunkStore.addChunksForChapter(doc.metadata.chapter, doc.pageContent);
        this.attachAutoUpdate(doc);
    }

    async getChapter(chapter: number): Promise<ChapterDocument | undefined> {
        await this.loadPromise;
        const result = this.collection.findOne({ 'metadata.chapter': chapter });
        return result ? this.attachAutoUpdate(result) as ChapterDocument : undefined;
    }

    async getChapterSummary(chapter: number): Promise<ChapterSummaryDocument | undefined> {
        return this.summaryStore.getChapterSummary(chapter);
    }

    async getChapterChunks(chapter: number): Promise<ChapterChunkDocument[]> {
        return this.chapterChunkStore.getChapterChunks(chapter);
    }

    async close(): Promise<void> {
        await promisify(this.db.saveDatabase.bind(this.db))();
        // Do not close the Loki instance here because it's shared.
    }
}
