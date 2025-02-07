import { Document } from '@langchain/core/documents';
import type { ChapterDocument } from './ChapterDocumentStore';
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
    private isClosed = false;
    private autoUpdateTimers = new Set<ReturnType<typeof setTimeout>>();
    // Removed collection; using this.db directly.

    constructor(config: { db: PouchDB.Database, debugBar?: MultiBar }) {
        this.debugBar = config.debugBar;
        this.db = config.db;
        (async () => {
            await this.db.createIndex({ index: { fields: ['metadata.docType', 'metadata.novelID', 'metadata.chapter'] } });
        })();
    }

    async addChapterSummary(doc: ChapterSummaryDocument): Promise<void> {
        if(!doc.metadata || !_.isNumber(doc.metadata.chapter)) {
            throw new Error('chapter metadata is required');
        }
        // Mimic ChapterChunkDocumentStore:
        const summaryDoc = doc as ChapterSummaryDocument & { _id?: string };
        summaryDoc._id = `chapter_summary_${doc.metadata.novelID}_${doc.metadata.chapter}`;
        await this.db.put(doc);
    }

    async getChapterSummary(chapter: ChapterDocument): Promise<ChapterSummaryDocument | undefined> {
        // eslint-disable-next-line lodash/prefer-lodash-method -- not actually an array
        const res = await this.db.find({
            selector: { 'metadata.docType': CHAPTER_SUMMARY_DOCTYPE, 'metadata.chapter': chapter.metadata.chapter }
        }) as unknown as { docs: ChapterSummaryDocument[] };
        return res?.docs[0];
    }
}
