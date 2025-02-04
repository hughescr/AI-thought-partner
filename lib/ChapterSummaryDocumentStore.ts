import { Document } from '@langchain/core/documents';
import Loki from 'lokijs';
import _ from 'lodash';
import { promisify } from 'node:util';

export class ChapterSummaryDocument extends Document<{ chapter: number }> {
    constructor(fields: { pageContent: string, metadata: { chapter: number } }) {
        super(fields);
    }
}

export class ChapterSummaryDocumentStore {
    private db: Loki;
    private collection: Loki.Collection;

    constructor(db: Loki) {
        this.db = db;
        this.collection =
            this.db.getCollection('summaries') ||
            this.db.addCollection('summaries', {
                unique: ['metadata.novelID', 'metadata.chapter'],
                indices: ['metadata.novelID', 'metadata.chapter']
            });
    }

    // Helper function: attaches an auto-update hook to the pageContent property.
    // When pageContent is modified, the document is updated and the db is saved.
    private attachAutoUpdate(doc: ChapterSummaryDocument): ChapterSummaryDocument {
        let currentContent = doc.pageContent;
        Object.defineProperty(doc, 'pageContent', {
            get: () => currentContent,
            set: (newVal) => {
                currentContent = newVal;
                this.collection.update(doc);
                // Explicitly save immediately and log potential errors
                promisify(this.db.saveDatabase.bind(this.db))().catch(err => {
                    console.error(`Error saving summary document:`, err);
                });
            },
            configurable: true
        });
        return doc;
    }

    async addChapterSummary(doc: ChapterSummaryDocument): Promise<void> {
        if(!doc.metadata || !_.isNumber(doc.metadata.chapter)) {
            throw new Error('chapter metadata is required');
        }
        this.collection.insert(doc);
        await promisify(this.db.saveDatabase.bind(this.db))();
        // Attach auto-update hook to the document so property changes get persisted
        this.attachAutoUpdate(doc);
    }

    async getChapterSummary(chapter: number): Promise<ChapterSummaryDocument | undefined> {
        const result = this.collection.findOne({ 'metadata.chapter': chapter });
        if(result) {
            // Attach auto-update hook on the persisted document and return it directly.
            return this.attachAutoUpdate(result) as ChapterSummaryDocument;
        }
        return undefined;
    }

    // New close method to properly shut down the database connection
    async close(): Promise<void> {
        await promisify(this.db.saveDatabase.bind(this.db))();
        // Do not close the Loki instance here because it's shared.
    }
}
