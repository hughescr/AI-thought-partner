import { TextSplitter } from 'langchain/text_splitter';
import { Document } from 'langchain/document';

/**
 * Splits text based on Markdown chapter headers.
 * Any markdown header (e.g., # Chapter 1, ## Prologue) is considered a chapter boundary.
 * Text before the first chapter header is treated as a separate chunk.
 */
export class MarkdownChapterTextSplitter extends TextSplitter {
    private splitterRegex: RegExp;

    constructor() {
        super();
        // Matches any markdown header like '# Chapter 1', '## Prologue', etc.
        this.splitterRegex = /^#+\s+.+$/gm;
    }

    /**
     * Splits the input text into chunks based on markdown headers.
     * @param text - The text to be split.
     * @returns An array of text chunks.
     */
    async splitText(text: string): Promise<string[]> {
        const splits: string[] = [];
        let lastIndex = 0;
        let match: RegExpExecArray | null;

        // Iterate over all header matches
        while ((match = this.splitterRegex.exec(text)) !== null) {
            const matchIndex = match.index;
            if (matchIndex > lastIndex) {
                const chunk = text.slice(lastIndex, matchIndex).trim();
                if (chunk) {
                    splits.push(chunk);
                }
            }
            // Advance lastIndex to the start of the matched header
            lastIndex = matchIndex;
        }

        // Add any remaining text after the last header
        if (lastIndex < text.length) {
            const chunk = text.slice(lastIndex).trim();
            if (chunk) {
                splits.push(chunk);
            }
        }

        return splits;
    }

    /**
     * Splits an array of documents into smaller chunks based on markdown headers.
     * @param documents - The documents to be split.
     * @returns An array of split documents.
     */
    async splitDocuments(documents: Document[]): Promise<Document[]> {
        const newDocs: Document[] = [];
        for (const doc of documents) {
            const chunks = await this.splitText(doc.pageContent);
            for (const chunk of chunks) {
                newDocs.push(new Document({ pageContent: chunk }));
            }
        }
        return newDocs;
    }
}
