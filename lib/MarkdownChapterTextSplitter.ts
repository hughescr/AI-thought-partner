import { TextSplitter } from 'langchain/text_splitter';
import { Document } from 'langchain/document';
import _ from 'lodash';

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
        this.splitterRegex = /^#+\s+\S.*$/gm;
    }

    /**
     * Splits the input text into chunks based on markdown headers.
     * @param text - The text to be split.
     * @returns An array of text chunks.
     */
    async splitText(text: string): Promise<string[]> {
        const splits: string[] = [];
        const matches = [];
        let match: RegExpExecArray | null;
        while((match = this.splitterRegex.exec(text)) !== null) {
            matches.push(match);
        }

        if(matches.length === 0) {
            const trimmed = _.trim(text);
            if (trimmed.length > 0) {
                splits.push(trimmed);
            }
            return splits;
        }

        const firstMatch = matches[0];
        if(firstMatch.index > 0) {
            const beforeHeader = _.trim(text.slice(0, firstMatch.index));
            if(beforeHeader) {
                splits.push(beforeHeader);
            }
        }

        for(let i = 0; i < matches.length; i++) {
            const currentMatch = matches[i];
            const start = currentMatch.index;
            let end: number;

            if(i < matches.length - 1) {
                end = matches[i + 1].index;
            } else {
                end = text.length;
            }

            const chunk = _.trim(text.slice(start, end));
            if(chunk) {
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
        for(const doc of documents) {
            const chunks = await this.splitText(doc.pageContent);
            for(const chunk of chunks) {
                newDocs.push(new Document({ pageContent: chunk }));
            }
        }
        return newDocs;
    }
}
