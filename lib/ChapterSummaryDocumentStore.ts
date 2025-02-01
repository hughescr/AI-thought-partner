import { Document } from '@langchain/core/documents';
import Loki from 'lokijs';
import { ensureFileSync } from 'fs-extra';

export class ChapterSummaryDocument extends Document<{ chapter: number }> {
    constructor(fields: { pageContent: string, metadata: { chapter: number } }) {
        super(fields);
    }
}

export class ChapterSummaryDocumentStore {
    private db: Loki;
    private collection: Loki.Collection;

    constructor(filePath: string) {
        ensureFileSync(filePath); // Create file if it doesn't exist
        this.db = new Loki(filePath, {
            adapter: new Loki.LokiFsAdapter(),
            autoload: true,
            autosave: true,
            autosaveInterval: 5000
        });

        this.collection = this.db.addCollection('summaries', {
            unique: ['metadata.chapter'],
            indices: ['metadata.chapter']
        });
    }

    addChapterSummary(doc: ChapterSummaryDocument): void {
        this.collection.insert(doc);
    }

    getChapterSummary(chapter: number): ChapterSummaryDocument | undefined {
        const result = this.collection.findOne({ 'metadata.chapter': chapter });
        return result
            ? new ChapterSummaryDocument({
                pageContent: result.pageContent,
                metadata: { chapter: result.metadata.chapter }
            })
            : undefined;
    }
}
