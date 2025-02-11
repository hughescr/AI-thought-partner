import { Document } from '@langchain/core/documents';
import PouchDB from 'pouchdb';
import find from 'pouchdb-find';
PouchDB.plugin(find);
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import _ from 'lodash';
import { ChapterDocument } from './ChapterDocumentStore';
import type { MultiBar } from 'cli-progress';

const CHAPTER_CHUNK_DOCTYPE = 'chapter_chunk';

export class ChapterChunkDocument extends Document<{ chapter: number, sequence: number, novelID: string, docType: string }> {
    constructor(fields: {
        pageContent: string
        metadata: { chapter: number, sequence: number, novelID: string }
    }) {
        super({
            ...fields,
            metadata: {
                ...fields.metadata,
                docType: CHAPTER_CHUNK_DOCTYPE,
            },
        });
    }
}

export class ChapterChunkDocumentStore {
    private debugBar?: MultiBar;
    private db!: PouchDB.Database;
    private initializationPromise: Promise<void>;
    private textSplitter: { splitText(text: string): Promise<string[]> };

    constructor(config: { db: PouchDB.Database, debugBar?: MultiBar, textSplitter?: { splitText(text: string): Promise<string[]> } }) {
        this.debugBar = config.debugBar;
        this.db = config.db;
        this.textSplitter = config.textSplitter ?? new RecursiveCharacterTextSplitter({
            chunkSize: 512,
            chunkOverlap: 128,
            keepSeparator: true
        });
        this.initializationPromise = this.db
            .createIndex({ index: { fields: ['metadata.docType', 'metadata.novelID', 'metadata.chapter', 'metadata.sequence'] } })
            .then(_.noop);
    }

    async deleteChapterChunks(chapterDoc: ChapterDocument): Promise<void> {
        await this.initializationPromise;
        const chapter = chapterDoc.metadata.chapter;
        const novelID = chapterDoc.metadata.novelID;
        // eslint-disable-next-line lodash/prefer-lodash-method -- PouchDB API
        const res = await this.db.find({
            selector: {
                metadata: {
                    docType: CHAPTER_CHUNK_DOCTYPE,
                    chapter,
                    novelID,
                },
            },
            limit: Number.MAX_SAFE_INTEGER,
        });

        if(res.docs.length === 0) {
            return;
        }

        await Promise.all(_.map(res.docs, async (doc) => {
            const freshDoc = await this.db.get(doc._id!);
            return this.db.remove(freshDoc);
        }));
    }

    async addChunksForChapter(chapterDoc: ChapterDocument): Promise<void> {
        await this.initializationPromise;
        const chapter = chapterDoc.metadata.chapter;
        const novelID = chapterDoc.metadata.novelID;
        const content = chapterDoc.pageContent;

        const chunks = await this.textSplitter.splitText(content);
        if(!chunks.length) {
            throw new Error('No text chunks generated');
        }
        const bar = this.debugBar?.create(chunks.length, 0, { msg: `${_.chain(content).split('\n').head().trim().value()} chunks` });
        const chunkDocs = await Promise.all(_.map(chunks, async (chunk, sequence) => {
            bar?.increment();
            const chunkDoc = new ChapterChunkDocument({
                pageContent: chunk,
                metadata: {
                    chapter,
                    sequence: sequence + 1,
                    novelID
                }
            }) as ChapterChunkDocument & { _id: string };
            chunkDoc._id = `chapter_chunk_${novelID}_${chapter}_${sequence + 1}`;
            return chunkDoc;
        }));
        // Put them all
        const result = await this.db.bulkDocs(chunkDocs);
        _.forEach(result, (res) => {
            if('error' in res && res.error) {
                throw res;
            }
        });
        bar?.stop();
        if(bar) {
            this.debugBar?.remove(bar);
        }
    }

    async getChapterChunks(chapterDoc: ChapterDocument): Promise<ChapterChunkDocument[]> {
        await this.initializationPromise;
        const chapter = chapterDoc.metadata.chapter;
        const novelID = chapterDoc.metadata.novelID;
        // eslint-disable-next-line lodash/prefer-lodash-method -- PouchDB API
        const res = await this.db.find({
            selector: {
                metadata: {
                    docType: CHAPTER_CHUNK_DOCTYPE,
                    chapter,
                    novelID,
                },
            },
            limit: Number.MAX_SAFE_INTEGER,
        });
        return res.docs as unknown as ChapterChunkDocument[];
    }
}
