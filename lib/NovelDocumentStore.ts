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

import type { MultiBar, SingleBar } from 'cli-progress';

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
    private debugBar?: MultiBar;
    private novelsCollection: Loki.Collection;
    private chapterStore: ChapterDocumentStore;
    private chapterSplitter: MarkdownChapterTextSplitter;
    private filePath: string;
    public embeddings: Embeddings;
    private faissStore!: Promise<FaissStoreWithMMR>;
    // eslint-disable-next-line lodash/prefer-constant -- Cannot use _.constant here because we need to declare this before calling super()
    public _vectorstoreType() { return 'novel'; }

    constructor(embeddings: Embeddings, dbConfig: { filePath: string, summaryGenerator: ChapterSummaryGenerator, debugBar?: MultiBar }) {
        super(embeddings, dbConfig);
        this.debugBar = dbConfig.debugBar;
        this.db = new Loki(dbConfig.filePath, {
            adapter: new Loki.LokiFsAdapter(),
            autoload: true,
            autosave: true,
            autosaveInterval: 200,
            throttledSaves: true,
        });
        this.novelsCollection =
            this.db.getCollection('novels') ||
            this.db.addCollection('novels', {
                unique: ['metadata.novelID'],
                indices: ['metadata.novelID'],
                autoupdate: true,
            });
        // Create our own ChapterDocumentStore using the same Loki instance.
        this.filePath = `${dbConfig.filePath}-FAISS`;
        this.embeddings = embeddings;
        this.chapterStore = new ChapterDocumentStore({ db: this.db, summaryGenerator: dbConfig.summaryGenerator, debugBar: this.debugBar });
        this.chapterSplitter = new MarkdownChapterTextSplitter();
        this.faissStore = FaissStore.load(this.filePath, this.embeddings).catch(() => new FaissStore(this.embeddings, {})) as Promise<FaissStoreWithMMR>;
    }

    async addNovel(novelDoc: NovelDocument): Promise<void> {
        // Persist the novel document.
        await this.persistNovel(novelDoc);

        // Split the novel into chapters.
        const chapters = await this.chapterSplitter.splitDocuments([novelDoc]);
        const chapterBar = this.debugBar?.create(chapters.length, 0, { msg: 'Chapters' });
        const faiss = await this.faissStore;

        // Process each chapter using a helper.
        for(let i = 0; i < chapters.length; i++) {
            await this.processChapter(chapters[i], i, novelDoc.metadata.novelID!, faiss, chapterBar);
        }

        chapterBar?.stop();
        if(chapterBar) {
            this.debugBar?.remove(chapterBar);
        }
    }

    private async persistNovel(novelDoc: NovelDocument): Promise<void> {
        // Ensure novelID is set (compute reproducibly if missing) and insert the document.
        if(!novelDoc.metadata.novelID) {
            novelDoc.metadata.novelID = computeNovelID(novelDoc.metadata.author, novelDoc.metadata.title);
        }
        this.novelsCollection.insert(novelDoc);
        await new Promise<void>((resolve, reject) => {
            this.db.saveDatabase(err => (err ? reject(err) : resolve()));
        });
    }

    private async processChapter(
        chapter: Document,
        index: number,
        novelID: string,
        faiss: FaissStoreWithMMR,
        chapterBar: SingleBar
    ): Promise<void> {
        const chapterTitle = this.extractChapterTitle(chapter.pageContent);
        chapterBar?.update(index + 1, { msg: chapterTitle });

        this.updateChapterMetadata(chapter, index + 1, novelID);

        const chapterDoc = new ChapterDocument({
            pageContent: chapter.pageContent,
            metadata: { novelID, chapter: index + 1 },
        });
        await this.chapterStore.addChapter(chapterDoc);

        await this.processChapterSummary(chapterTitle, index + 1, faiss, chapterBar);
        await this.processChapterChunks(chapterTitle, index + 1, faiss, chapterBar);

        if(faiss._index) {
            await faiss.save(this.filePath);
        }
    }

    private extractChapterTitle(pageContent: string): string {
        return _.chain(pageContent).split('\n').head().trim().value();
    }

    private updateChapterMetadata(chapter: Document, chapterNum: number, novelID: string): void {
        if(!chapter.metadata) {
            chapter.metadata = {};
        }
        chapter.metadata.chapter = chapterNum;
        chapter.metadata.novelID = novelID;
    }

    private async processChapterSummary(
        chapterTitle: string,
        chapterNum: number,
        faiss: FaissStoreWithMMR,
        chapterBar: SingleBar
    ): Promise<void> {
        const summaryDoc = await this.chapterStore.getChapterSummary(chapterNum);
        if(summaryDoc) {
            chapterBar?.update(chapterNum, { msg: `Adding summary of ${chapterTitle} to FAISS` });
            await faiss.addDocuments([summaryDoc]);
        }
    }

    private async processChapterChunks(
        chapterTitle: string,
        chapterNum: number,
        faiss: FaissStoreWithMMR,
        chapterBar: SingleBar
    ): Promise<void> {
        chapterBar?.update(chapterNum, { msg: `Getting chunks of ${chapterTitle}` });
        const chunks = await this.chapterStore.getChapterChunks(chapterNum);
        chapterBar?.update(chapterNum, { msg: `Got ${chunks.length} chunks of ${chapterTitle}` });
        if(chunks.length > 0) {
            const chunkBar = this.debugBar?.create(chunks.length, 0, { msg: `Adding chunks of ${chapterTitle} to FAISS` });
            for(const chunk of chunks) {
                chunkBar?.increment();
                await faiss.addDocuments([chunk]);
            }
            chunkBar?.stop();
            if(chunkBar) {
                this.debugBar?.remove(chunkBar);
            }
        }
        chapterBar?.update(chapterNum, { msg: `Done with chunks of ${chapterTitle}` });
    }

    async addDocuments(documents: Document[]): Promise<void> {
        for(const doc of documents) {
            if(!doc.metadata || !doc.metadata.title || !doc.metadata.author) {
                throw new Error('Document missing required novel metadata');
            }
            await this.addNovel(doc as NovelDocument);
        }
    }

    async addVectors(_vectors: number[][], _documents: Document[]): Promise<string[]> {
        throw new Error('Method not implemented. You can add documents via addDocuments or addNovel.');
    }

    async similaritySearchVectorWithScore(query: number[], k: number): Promise<[Document, number][]> {
        const faissResults = await (await this.faissStore).similaritySearchVectorWithScore(query, k);
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
        const faiss = await this.faissStore;
        // Only save if something was added to the index.
        if(faiss._index) {
            await faiss.save(this.filePath);
        }
        return this.db.close();
    }
}
