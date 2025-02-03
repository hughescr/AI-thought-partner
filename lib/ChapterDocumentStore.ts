import { Document } from '@langchain/core/documents';
import Loki from 'lokijs';
import _ from 'lodash';
import { promisify } from 'node:util';
import { ChapterSummaryDocumentStore, ChapterSummaryDocument } from './ChapterSummaryDocumentStore';
import { ChapterSummaryGenerator } from './ChapterSummaryGenerator';

export class ChapterDocument extends Document<{ novelID: string, chapter: number }> {
    constructor(fields: { pageContent: string, metadata: { novelID: string, chapter: number } }) {
        super(fields);
    }
}

export class ChapterDocumentStore {
    private db!: Loki;
    private collection!: Loki.Collection;
    private summaryStore!: ChapterSummaryDocumentStore;
    private summaryGenerator: ChapterSummaryGenerator;
    private loadPromise: Promise<void>;

    constructor(db: Loki, summaryGenerator: ChapterSummaryGenerator) {
        this.summaryGenerator = summaryGenerator;
        this.db = db;
        this.loadPromise = Promise.resolve();
        // Create (or get) the chapters collection with unique compound index on novelID and chapter.
        this.collection =
            this.db.getCollection('chapters') ||
            this.db.addCollection('chapters', {
                unique: ['metadata.novelID', 'metadata.chapter'],
                indices: ['metadata.novelID', 'metadata.chapter']
            });
        // Pass the same db instance to the summary store.
        this.summaryStore = new ChapterSummaryDocumentStore(this.db);
    }

    private attachAutoUpdate(doc: ChapterDocument): ChapterDocument {
        let currentContent = doc.pageContent;
        Object.defineProperty(doc, 'pageContent', {
            get: () => currentContent,
            set: (newVal) => {
                currentContent = newVal;
                this.collection.update(doc);
                promisify(this.db.saveDatabase.bind(this.db))();
                console.log(`Chapter auto-update: pageContent updated for chapter ${doc.metadata.chapter}`);
                // Async update of chapter summary
                (async () => {
                    const startTimestamp = new Date().toISOString();
                    console.log(`Regenerating summary for chapter ${doc.metadata.chapter} at ${startTimestamp}`);
                    const genStart = Date.now();
                    // Regenerate the summary using updated pageContent
                    const summaryResult = await this.summaryGenerator.generateSummary(
                        new Document({ pageContent: newVal, metadata: doc.metadata })
                    );
                    const genEnd = Date.now();
                    console.log(`Summary generation completed for chapter ${doc.metadata.chapter} at ${new Date().toISOString()} (elapsed: ${genEnd - genStart}ms)`);
                    // Retrieve existing summary
                    const existingSummary = await this.summaryStore.getChapterSummary(doc.metadata.chapter);
                    if(existingSummary) {
                        console.log(`Updating existing summary for chapter ${doc.metadata.chapter} at ${new Date().toISOString()}`);
                        // Update existing summary
                        existingSummary.pageContent = summaryResult.pageContent;
                        const saveStart = Date.now();
                        await promisify(this.db.saveDatabase.bind(this.db))();
                        const saveEnd = Date.now();
                        console.log(`Database save completed for chapter ${doc.metadata.chapter} at ${new Date().toISOString()} (elapsed: ${saveEnd - saveStart}ms)`);
                    } else {
                        console.log(`Adding new summary for chapter ${doc.metadata.chapter} at ${new Date().toISOString()}`);
                        // Add new summary if not present
                        summaryResult.metadata.chapter = doc.metadata.chapter;
                        const addStart = Date.now();
                        await this.summaryStore.addChapterSummary(summaryResult as ChapterSummaryDocument);
                        const addEnd = Date.now();
                        console.log(`Summary addition completed for chapter ${doc.metadata.chapter} at ${new Date().toISOString()} (elapsed: ${addEnd - addStart}ms)`);
                    }
                })();
            },
            configurable: true
        });
        return doc;
    }

    private async generateAndStoreSummary(doc: ChapterDocument): Promise<void> {
        const summaryDoc = await this.summaryGenerator.generateSummary(doc);
        summaryDoc.metadata.chapter = doc.metadata.chapter;
        console.log(`Generated summary for chapter ${doc.metadata.chapter}`);
        await this.summaryStore.addChapterSummary(summaryDoc as ChapterSummaryDocument);
    }

    async addChapter(doc: ChapterDocument): Promise<void> {
        if(!doc.metadata || !_.isNumber(doc.metadata.chapter) || !doc.metadata.novelID) {
            throw new Error('chapter metadata is required');
        }
        if(this.collection.findOne({ 'metadata.novelID': doc.metadata.novelID, 'metadata.chapter': doc.metadata.chapter })) {
            console.log(`Duplicate add attempted for chapter ${doc.metadata.chapter}`);
            throw new Error('Document is already in collection, please use update()');
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

    async getChapterSummary(chapter: number): Promise<Document | undefined> {
        return this.summaryStore.getChapterSummary(chapter);
    }

    async close(): Promise<void> {
        await this.loadPromise;
        await promisify(this.db.saveDatabase.bind(this.db))();
        console.log('Database closed for ChapterDocumentStore');
        this.db.close();
    }
}
