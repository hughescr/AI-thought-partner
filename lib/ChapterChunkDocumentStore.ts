import { Document } from '@langchain/core/documents';
import Loki from 'lokijs';
import { promisify } from 'node:util';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import _ from 'lodash';

export class ChapterChunkDocument extends Document<{ chapter: number, sequence: number }> {
    constructor(fields: {
        pageContent: string
        metadata: { chapter: number, sequence: number }
    }) {
        super(fields);
    }
}

export class ChapterChunkDocumentStore {
    private db: Loki;
    private collection: Loki.Collection;
    private textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 512,
        chunkOverlap: 128,
        keepSeparator: true
    });

    constructor(db: Loki) {
        this.db = db;
        this.collection = this.db.getCollection('chapter_chunks') || this.db.addCollection('chapter_chunks', {
            unique: ['metadata.novelID', 'metadata.chapter', 'metadata.sequence'],
            indices: ['metadata.novelID', 'metadata.chapter', 'metadata.sequence']
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

        try {
            _.forEach(chunks, (chunkContent, sequence) => {
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
