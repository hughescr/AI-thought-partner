import { Document } from '@langchain/core/documents';
import Loki from 'lokijs';
import _ from 'lodash';
import { promisify } from 'node:util';
import { ChapterSummaryDocumentStore, ChapterSummaryDocument } from './ChapterSummaryDocumentStore';
import { ChapterSummaryGenerator } from './ChapterSummaryGenerator';

export class ChapterDocument extends Document<{ chapter: number }> {
    constructor(fields: { pageContent: string, metadata: { chapter: number } }) {
        super(fields);
    }
    async getChapterSummary(chapter: number): Promise<Document | undefined> {
        return this.summaryStore.getChapterSummary(chapter);
    }
}

export class ChapterDocumentStore {
    private db!: Loki;
    private collection!: Loki.Collection;
    private loadPromise: Promise<void>;

    constructor(filePath: string, summaryGenerator: ChapterSummaryGenerator) {
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
                    this.summaryStore = new ChapterSummaryDocumentStore(this.db);
                    this.summaryGenerator = summaryGenerator;
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

    private async generateAndStoreSummary(doc: ChapterDocument): Promise<void> {
        const summaryContent = await this.summaryGenerator.generateSummary(
            new Document({ pageContent: doc.pageContent, metadata: doc.metadata })
        );
        const summaryDoc = new ChapterSummaryDocument({
            pageContent: summaryContent.pageContent,
            metadata: { chapter: doc.metadata.chapter }
        });
        await this.summaryStore.addChapterSummary(summaryDoc);
    }

    async addChapter(doc: ChapterDocument): Promise<void> {
        await this.loadPromise;
        if(!doc.metadata || !_.isNumber(doc.metadata.chapter)) {
            throw new Error('chapter metadata is required');
        }
        this.collection.insert(doc);
        await promisify(this.db.saveDatabase.bind(this.db))();
        await this.generateAndStoreSummary(doc);
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
