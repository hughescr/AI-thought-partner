import { Document } from '@langchain/core/documents';
import PouchDB from 'pouchdb';
import find from 'pouchdb-find';
PouchDB.plugin(find);
import { ChapterSummaryDocumentStore, ChapterSummaryDocument } from './ChapterSummaryDocumentStore';
import { ChapterChunkDocumentStore, ChapterChunkDocument } from './ChapterChunkDocumentStore';
import type { ChapterSummaryGenerator } from './ChapterSummaryGenerator';

import type { MultiBar } from 'cli-progress';
import _ from 'lodash';
import { NovelDocument } from './NovelDocumentStore';

const CHAPTER_DOCTYPE = 'chapter';

export class ChapterDocument extends Document<{ novelID: string, chapter: number, docType: string }> {
    constructor(fields: { pageContent: string, metadata: { novelID: string, chapter: number } }) {
        super({
            ...fields,
            metadata: {
                ...fields.metadata,
                docType: CHAPTER_DOCTYPE,
            },
        });
    }
}

export class ChapterDocumentStore {
    private db!: PouchDB.Database;
    private debugBar?: MultiBar;
    private summaryStore!: ChapterSummaryDocumentStore;
    private chapterChunkStore!: ChapterChunkDocumentStore;
    private initializationPromise: Promise<void>;

    constructor(config: { db: PouchDB.Database, summaryGenerator?: ChapterSummaryGenerator, debugBar?: MultiBar }) {
        this.debugBar = config.debugBar;
        this.db = config.db;
        this.initializationPromise = this.db.createIndex({
            index: { fields: ['metadata.docType', 'metadata.novelID', 'metadata.chapter'] }
        }).then(_.noop);
        // Pass the same db instance to the summary store.
        this.summaryStore = new ChapterSummaryDocumentStore({ db: this.db, summaryGenerator: config.summaryGenerator, debugBar: this.debugBar });
        this.chapterChunkStore = new ChapterChunkDocumentStore({ db: this.db, debugBar: this.debugBar });
    }

    async addChapter(doc: ChapterDocument): Promise<void> {
        await this.initializationPromise;
        if(!doc.metadata || !_.isNumber(doc.metadata.chapter) || !doc.metadata.novelID) {
            throw new Error('chapter metadata is required');
        }
        const storeDoc = doc as ChapterDocument & { _id: string };
        storeDoc._id = `chapter_${doc.metadata.novelID}_${doc.metadata.chapter}`;
        
        // Check if document already exists
        try {
            await this.db.get(storeDoc._id);
            throw new Error('Document is already in collection, please use update()');
        } catch (err: any) {
            // If document doesn't exist (404), continue with adding it
            if (err.status !== 404) {
                throw err; // Re-throw if it's not a "not found" error
            }
        }
        
        await this.db.put(storeDoc);
        // No explicit save here—autosave will handle it.
        await this.summaryStore.addChapterSummary(doc);
        await this.chapterChunkStore.addChunksForChapter(doc);
    }

    async getChapter(novel: NovelDocument, chapter: number): Promise<ChapterDocument | undefined> {
        await this.initializationPromise;
        // eslint-disable-next-line lodash/prefer-lodash-method -- not actually an array
        const res = await this.db.find({
            selector: {
                metadata: {
                    docType: CHAPTER_DOCTYPE,
                    novelID: novel.metadata.novelID,
                    chapter,
                },
            },
            limit: Number.MAX_SAFE_INTEGER,
        });
        return res.docs[0] as unknown as ChapterDocument;
    }

    async getChapters(novel: NovelDocument): Promise<ChapterDocument[]> {
        await this.initializationPromise;
        // eslint-disable-next-line lodash/prefer-lodash-method -- not actually an array
        const res = await this.db.find({
            selector: {
                metadata: {
                    docType: CHAPTER_DOCTYPE,
                    novelID: novel.metadata.novelID,
                },
            },
            limit: Number.MAX_SAFE_INTEGER,
        });
        return res.docs as unknown as ChapterDocument[];
    }

    async getChapterSummary(chapter: ChapterDocument): Promise<ChapterSummaryDocument | undefined> {
        return this.summaryStore.getChapterSummary(chapter);
    }

    async getChapterSummaries(novelID: string): Promise<ChapterSummaryDocument[]> {
        return this.summaryStore.getChapterSummaries(novelID);
    }

    async getChapterChunks(chapter: ChapterDocument): Promise<ChapterChunkDocument[]> {
        return this.chapterChunkStore.getChapterChunks(chapter);
    }
}
