import { Document } from '@langchain/core/documents';
import type { ChapterDocument } from './ChapterDocumentStore';
import type { ChapterSummaryGenerator } from './ChapterSummaryGenerator';
import PouchDB from 'pouchdb';
import find from 'pouchdb-find';
PouchDB.plugin(find);
import _ from 'lodash';
import type { MultiBar } from 'cli-progress';

const CHAPTER_SUMMARY_DOCTYPE = 'summary';

export class ChapterSummaryDocument extends Document<{ chapter: number, novelID: string, docType: string }> {
    constructor(fields: { pageContent: string, metadata: { chapter: number, novelID: string } }) {
        super({
            ...fields,
            metadata: {
                ...fields.metadata,
                docType: CHAPTER_SUMMARY_DOCTYPE,
            },
        });
    }
}

export class ChapterSummaryDocumentStore {
    private db!: PouchDB.Database;
    private debugBar?: MultiBar;
    private summaryGenerator?: ChapterSummaryGenerator;
    private isClosed = false;
    private autoUpdateTimers = new Set<ReturnType<typeof setTimeout>>();
    private initializationPromise: Promise<void>;

    constructor(config: { db: PouchDB.Database, summaryGenerator?: ChapterSummaryGenerator, debugBar?: MultiBar }) {
        this.debugBar = config.debugBar;
        this.db = config.db;
        this.summaryGenerator = config.summaryGenerator;
        this.initializationPromise = this.db.createIndex({
            index: { fields: ['metadata.docType', 'metadata.novelID', 'metadata.chapter'] }
        }).then(_.noop);
    }

    async addChapterSummary(chapterDoc: ChapterDocument): Promise<void> {
        await this.initializationPromise;
        const chapterTitle = _.chain(chapterDoc.pageContent).split('\n').head()?.trim().trim('#').trim().value();
        const bar = this.debugBar?.create(1, 0, { msg: `Generating summary for ${chapterTitle}` });
        const summaryDoc = await this.summaryGenerator!.generateSummary(chapterDoc);
        bar?.stop();
        if(bar) {
            this.debugBar?.remove(bar);
        }
        // Ensure proper metadata.
        summaryDoc.metadata.chapter = chapterDoc.metadata.chapter;
        summaryDoc.metadata.novelID = chapterDoc.metadata.novelID;
        summaryDoc.metadata.docType = 'summary';
        const sDoc = summaryDoc as ChapterSummaryDocument & { _id?: string };
        sDoc._id = `chapter_summary_${chapterDoc.metadata.novelID}_${chapterDoc.metadata.chapter}`;
        await this.db.put(sDoc);
    }

    async getChapterSummary(chapter: ChapterDocument): Promise<ChapterSummaryDocument | undefined> {
        await this.initializationPromise;
        // eslint-disable-next-line lodash/prefer-lodash-method -- not actually an array
        const res = await this.db.find({
            selector: {
                metadata: {
                    docType: CHAPTER_SUMMARY_DOCTYPE,
                    chapter: chapter.metadata.chapter,
                },
            },
            limit: Number.MAX_SAFE_INTEGER,
        }) as unknown as { docs: ChapterSummaryDocument[] };
        return res?.docs[0];
    }

    async getChapterSummaries(novelID: string): Promise<ChapterSummaryDocument[]> {
        await this.initializationPromise;
        // eslint-disable-next-line lodash/prefer-lodash-method -- not actually an array
        const res = await this.db.find({
            selector: {
                metadata: {
                    docType: CHAPTER_SUMMARY_DOCTYPE,
                    novelID,
                },
            },
            limit: Number.MAX_SAFE_INTEGER,
        }) as unknown as { docs: ChapterSummaryDocument[] };
        return res.docs;
    }
}
