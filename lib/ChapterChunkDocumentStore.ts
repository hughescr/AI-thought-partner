import { Document } from '@langchain/core/documents';
import Loki from 'lokijs';
import { promisify } from 'node:util';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import _ from 'lodash';
import type { MultiBar } from 'cli-progress';

export class ChapterChunkDocument extends Document<{ chapter: number, sequence: number }> {
    constructor(fields: {
        pageContent: string
        metadata: { chapter: number, sequence: number }
    }) {
        super(fields);
    }
}

export class ChapterChunkDocumentStore {
    private debugBar?: MultiBar;
    private db: Loki;
    private collection: Loki.Collection;
    private textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 512,
        chunkOverlap: 128,
        keepSeparator: true
    });

    constructor(config: { db: Loki, debugBar?: MultiBar }) {
        this.debugBar = config.debugBar;
        this.db = config.db;
        this.collection = this.db.getCollection('chapter_chunks') || this.db.addCollection('chapter_chunks', {
            unique: ['metadata.novelID', 'metadata.chapter', 'metadata.sequence'],
            indices: ['metadata.novelID', 'metadata.chapter', 'metadata.sequence'],
            autoupdate: true,
        });
    }

    async deleteChapterChapters(chapter: number): Promise<void> {
        this.collection.findAndRemove({ 'metadata.chapter': chapter });
        // Rely on autosave.
    }

    async addChunksForChapter(chapter: number, content: string): Promise<void> {
        if(this.collection.findOne({ 'metadata.chapter': chapter })) {
            throw new Error('Duplicate key for properties metadata.chapter, metadata.sequence');
        }
        const chunks = await this.textSplitter.splitText(content);
        const bar = this.debugBar?.create(chunks.length, 0, { msg: `${_.chain(content).split('\n').head().trim().value()} chunks` });
        try {
            _.forEach(chunks, (chunkContent, sequence) => {
                bar?.increment();
                this.collection.insert(new ChapterChunkDocument({
                    pageContent: chunkContent,
                    metadata: {
                        chapter,
                        sequence: sequence + 1 // Start sequences at 1
                    }
                }));
            });
        } catch(error) {
            if(_.isError(error) && error.message.includes('unique')) {
                throw new Error('Duplicate key for properties metadata.chapter, metadata.sequence');
            }
            throw error;
        }
        bar?.stop();
        if(bar) {
            this.debugBar?.remove(bar);
        }
        // No explicit save call, autosave handles it.
    }

    async getChapterChunks(chapter: number): Promise<ChapterChunkDocument[]> {
        // eslint-disable-next-line lodash/prefer-lodash-method -- not actually an array
        return this.collection
            .chain()
            .find({ 'metadata.chapter': chapter })
            .simplesort('metadata.sequence')
            .data() as ChapterChunkDocument[];
    }

    async close(): Promise<void> {
        await promisify(this.db.saveDatabase.bind(this.db))();
        // Do not close the Loki instance here because it's shared.
    }
}
