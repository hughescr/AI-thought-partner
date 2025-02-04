import { Document } from '@langchain/core/documents';
import { MarkdownChapterTextSplitter } from './MarkdownChapterTextSplitter';
import { ChapterDocumentStore } from './ChapterDocumentStore';
import { ChapterDocument } from './ChapterDocumentStore';
import type { ChapterSummaryGenerator } from './ChapterSummaryGenerator';
import _ from 'lodash';
import Loki from 'lokijs';

import { VectorStore } from '@langchain/core/vectorstores';
import { FaissStore } from '@langchain/community/vectorstores/faiss';
import { FaissStoreWithMMR } from './FAISSStoreWithMMR';
import type { Embeddings } from '@langchain/core/embeddings';

// NovelDocument: represents the complete novel in Markdown.
export class NovelDocument extends Document<{
    novelID?: string
    title: string
    author: string
    genre?: string
    filepath?: string
}> {
    constructor(fields: {
        pageContent: string
        metadata: { novelID?: string, title: string, author: string, genre?: string, filepath?: string }
    }) {
        super(fields);
    }
}

// Helper to compute novelID reproducibly from author and title.
export function computeNovelID(author: string, title: string): string {
    // Simple reproducible ID, e.g. lowercased with spaces replaced by underscores.
    return `${_.chain(author).trim().toLower().replace(/\s+/g, '_').value()}_${_.chain(title).trim().toLower().replace(/\s+/g, '_').value()}`;
}

// NovelDocumentStore: stores novels and auto-splits them into chapters.
export class NovelDocumentStore extends VectorStore {
    private db: Loki;
    private novelsCollection: Loki.Collection;
    private chapterStore: ChapterDocumentStore;
    private chapterSplitter: MarkdownChapterTextSplitter;
    private filePath: string;
    public embeddings: Embeddings;
    private faissStore!: FaissStoreWithMMR;
    // eslint-disable-next-line lodash/prefer-constant -- Cannot use _.constant here because we need to declare this before calling super()
    public _vectorstoreType() { return 'novel'; }

    constructor(embeddings: Embeddings, dbConfig: { filePath: string, summaryGenerator: ChapterSummaryGenerator }) {
        super(embeddings, dbConfig);
        this.db = new Loki(dbConfig.filePath, {
            adapter: new Loki.LokiFsAdapter(),
            autoload: true,
            autosave: true,
            autosaveInterval: 200,   // autosave every 200ms
        });
        this.novelsCollection =
            this.db.getCollection('novels') ||
            this.db.addCollection('novels', {
                unique: ['metadata.novelID'],
                indices: ['metadata.novelID'],
            });
        // Create our own ChapterDocumentStore using the same Loki instance.
        this.filePath = `${dbConfig.filePath}-FAISS`;
        this.embeddings = embeddings;
        this.chapterStore = new ChapterDocumentStore(this.db, dbConfig.summaryGenerator);
        this.chapterSplitter = new MarkdownChapterTextSplitter();
    }

    // Add a novel document and split it into chapters.
    async addNovel(novelDoc: NovelDocument): Promise<void> {
        // Ensure novelID is set (compute reproducibly if missing)
        if(!novelDoc.metadata.novelID) {
            novelDoc.metadata.novelID = computeNovelID(novelDoc.metadata.author, novelDoc.metadata.title);
        }
        this.novelsCollection.insert(novelDoc);
        await new Promise<void>((resolve, reject) => {
            this.db.saveDatabase(err => (err ? reject(err) : resolve()));
        });
        // Split the full novel text into chapters via MarkdownChapterTextSplitter.
        const chapters = await this.chapterSplitter.splitDocuments([novelDoc]);
        // For each chapter, set metadata.novelID and a chapter number.
        for(let i = 0; i < chapters.length; i++) {
            if(!chapters[i].metadata) {
                chapters[i].metadata = {};
            }
            chapters[i].metadata.chapter = i + 1;
            chapters[i].metadata.novelID = novelDoc.metadata.novelID;
            const chapterDoc = new ChapterDocument({
                pageContent: chapters[i].pageContent,
                metadata: { novelID: novelDoc.metadata.novelID, chapter: i + 1 },
            });
            await this.chapterStore.addChapter(chapterDoc);
            const summaryDoc = await this.chapterStore.getChapterSummary(i + 1);
            if(summaryDoc) {
                await this.initFaissStore();
                await this.faissStore.addDocuments([summaryDoc]);
            }
            const chunks = await this.chapterStore.getChapterChunks(i + 1);
            if(chunks.length) {
                await this.initFaissStore();
                await this.faissStore.addDocuments(chunks);
            }
        }
    }

    async addDocuments(documents: Document[]): Promise<void> {
        for(const doc of documents) {
            if(!doc.metadata || !doc.metadata.title || !doc.metadata.author) {
                throw new Error('Document missing required novel metadata');
            }
            await this.addNovel(doc as NovelDocument);
        }
    }

    private async initFaissStore(): Promise<void> {
        try {
            if(!this.faissStore) {
                this.faissStore = await FaissStoreWithMMR.load(this.filePath, this.embeddings) as FaissStoreWithMMR;
            }
        } catch{
            this.faissStore = new FaissStore(this.embeddings, {}) as FaissStoreWithMMR;
        }
    }

    async addVectors(_vectors: number[][], _documents: Document[]): Promise<string[]> {
        throw new Error('Method not implemented. You can add documents via addDocuments or addNovel.');
    }

    async similaritySearchVectorWithScore(query: number[], k: number): Promise<[Document, number][]> {
        await this.initFaissStore();
        const faissResults = await this.faissStore.similaritySearchVectorWithScore(query, k);
        const seenChapters = new Set<number>();
        const results: [Document, number][] = [];
        for(const [doc, score] of faissResults) {
            if(doc.metadata?.chapter) {
                const chapNum = doc.metadata.chapter;
                if(!seenChapters.has(chapNum)) {
                    const chapterDoc = await this.chapterStore.getChapter(chapNum);
                    if(chapterDoc) {
                        results.push([chapterDoc, score]);
                        seenChapters.add(chapNum);
                    }
                }
                continue;
            }
            // If no chapter metadata is present.
            throw new Error(`Chapter metadata missing in document: ${JSON.stringify(doc)}`);
        }
        return results;
    }

    // New close method to properly shut down the database connection
    async close(): Promise<void> {
        if(this.faissStore) {
            await this.faissStore.save(this.filePath);
        }
        return this.db.close();
    }
}
