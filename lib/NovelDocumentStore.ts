import { Document } from '@langchain/core/documents';
import { MarkdownChapterTextSplitter } from './MarkdownChapterTextSplitter';
import { ChapterDocumentStore } from './ChapterDocumentStore';
import { ChapterDocument } from './ChapterDocumentStore';
import _ from 'lodash';
import Loki from 'lokijs';

// NovelDocument: represents the complete novel in Markdown.
export class NovelDocument extends Document<{
    novelID: string
    title: string
    author: string
    genre?: string
    filepath?: string
}> {
    constructor(fields: {
        pageContent: string
        metadata: { novelID: string, title: string, author: string, genre?: string, filepath?: string }
    }) {
        super(fields);
    }
}

// Helper to compute novelID reproducibly from author and title.
export function computeNovelID(author: string, title: string): string {
    // Simple reproducible ID, e.g. lowercased with spaces replaced by underscores.
    return `${_.trim(author).toLowerCase().replace(/\s+/g, '_')}_${_.trim(title).toLowerCase().replace(/\s+/g, '_')}`;
}

// NovelDocumentStore: stores novels and auto-splits them into chapters.
export class NovelDocumentStore {
    private db: Loki;
    private novelsCollection: Loki.Collection;
    private chapterStore: ChapterDocumentStore;
    private chapterSplitter: MarkdownChapterTextSplitter;

    constructor(filePath: string, chapterStore: ChapterDocumentStore) {
        this.db = new Loki(filePath, {
            adapter: new Loki.LokiFsAdapter(),
            autoload: true,
            autosave: true,
            autosaveInterval: 5000,
        });
        this.novelsCollection =
      this.db.getCollection('novels') ||
      this.db.addCollection('novels', {
          unique: ['metadata.novelID'],
          indices: ['metadata.novelID'],
      });
        // Use the provided ChapterDocumentStore (which is created with the same Loki db)
        this.chapterStore = chapterStore;
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
        }
    }
}
