import { Document } from '@langchain/core/documents';
import Loki from 'lokijs';
import _ from 'lodash';
import { promisify } from 'node:util';

export class ChapterDocument extends Document<{ chapter: number }> {
    constructor(fields: { pageContent: string, metadata: { chapter: number } }) {
        super(fields);
    }
}

export class ChapterDocumentStore {
    private db!: Loki;
    private collection!: Loki.Collection;
    private loadPromise: Promise<void>;

    constructor(filePath: string) {
        this.loadPromise = new Promise((resolve) => {
            this.db = new Loki(filePath, {
                adapter: new Loki.LokiFsAdapter(),
                autoload: true,
                autoloadCallback: () => {
                    this.collection =
                        this.db.getCollection('chapters') ||
                        this.db.addCollection('chapters', {
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

    private attachAutoUpdate(doc: ChapterDocument): ChapterDocument {
        let currentContent = doc.pageContent;
        Object.defineProperty(doc, 'pageContent', {
            get: () => currentContent,
            set: (newVal) => {
                currentContent = newVal;
                this.collection.update(doc);
                promisify(this.db.saveDatabase.bind(this.db))();
            },
            configurable: true
        });
        return doc;
    }

    async addChapter(doc: ChapterDocument): Promise<void> {
        await this.loadPromise;
        if(!doc.metadata || !_.isNumber(doc.metadata.chapter)) {
            throw new Error('chapter metadata is required');
        }
        this.collection.insert(doc);
        await promisify(this.db.saveDatabase.bind(this.db))();
        this.attachAutoUpdate(doc);
    }

    async getChapter(chapter: number): Promise<ChapterDocument | undefined> {
        await this.loadPromise;
        const result = this.collection.findOne({ 'metadata.chapter': chapter });
        return result ? this.attachAutoUpdate(result) as ChapterDocument : undefined;
    }

    async close(): Promise<void> {
        await this.loadPromise;
        await promisify(this.db.saveDatabase.bind(this.db))();
        this.db.close();
    }
}
