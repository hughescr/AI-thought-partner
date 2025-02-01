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
    private db!: Loki;
    private collection!: Loki.Collection;
    private loadPromise: Promise<void>;

    constructor(filePath: string) {
        // Wrap database creation and loading in a promise to ensure autoload completes
        this.loadPromise = new Promise((resolve) => {
            this.db = new Loki(filePath, {
                adapter: new Loki.LokiFsAdapter(),
                autoload: true,
                autoloadCallback: () => {
                    this.collection =
                        this.db.getCollection('summaries') ||
                        this.db.addCollection('summaries', {
                            unique: ['metadata.chapter'],
                            indices: ['metadata.chapter']
                        });
                    resolve();
                },
                autosave: true,
                autosaveInterval: 5000
            });
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
                // Fire-and-forget save (errors are ignored)
                promisify(this.db.saveDatabase.bind(this.db))();
            },
            configurable: true
        });
        return doc;
    }

    async addChapterSummary(doc: ChapterSummaryDocument): Promise<void> {
        await this.loadPromise; // Ensure DB is loaded
        if(!doc.metadata || !_.isNumber(doc.metadata.chapter)) {
            throw new Error('chapter metadata is required');
        }
        this.collection.insert(doc);
        await promisify(this.db.saveDatabase.bind(this.db))();
        // Attach auto-update hook to the document so property changes get persisted
        this.attachAutoUpdate(doc);
    }

    async getChapterSummary(chapter: number): Promise<ChapterSummaryDocument | undefined> {
        await this.loadPromise; // Ensure DB is loaded
        const result = this.collection.findOne({ 'metadata.chapter': chapter });
        if(result) {
            // Attach auto-update hook on the persisted document and return it directly.
            return this.attachAutoUpdate(result) as ChapterSummaryDocument;
        }
        return undefined;
    }

    // New close method to properly shut down the database connection
    async close(): Promise<void> {
        await this.loadPromise; // Ensure DB is loaded
        await promisify(this.db.saveDatabase.bind(this.db))();
        this.db.close();
    }
}
