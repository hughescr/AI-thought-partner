import { MarkdownChapterTextSplitter } from '../lib/MarkdownChapterTextSplitter';
import { Document } from 'langchain/document';

describe('MarkdownChapterTextSplitter', () => {
    let splitter: MarkdownChapterTextSplitter;

    beforeAll(() => {
        splitter = new MarkdownChapterTextSplitter();
    });

    test('should split text into chapters based on markdown headers', async () => {
        const text = `
# Chapter 1
This is the content of chapter one.

## Chapter 2
This is the content of chapter two.

### Chapter 3
This is the content of chapter three.
        `;
        const expectedChunks = [
            'This is the content of chapter one.',
            'This is the content of chapter two.',
            'This is the content of chapter three.',
        ];

        const result = await splitter.splitText(text);
        expect(result).toEqual(expectedChunks);
    });

    test('should recognize headers of different levels as chapter boundaries', async () => {
        const text = `
# Chapter 1
Content for chapter 1.

## Chapter 2
Content for chapter 2.

### Chapter 3
Content for chapter 3.

#### Chapter 4
Content for chapter 4.
        `;
        const expectedChunks = [
            'Content for chapter 1.',
            'Content for chapter 2.',
            'Content for chapter 3.',
            'Content for chapter 4.',
        ];

        const result = await splitter.splitText(text);
        expect(result).toEqual(expectedChunks);
    });

    test('should treat text before the first header as a separate chunk', async () => {
        const text = `
Introduction text before any chapter header.

# Chapter 1
Content for chapter 1.
        `;
        const expectedChunks = [
            'Introduction text before any chapter header.',
            'Content for chapter 1.',
        ];

        const result = await splitter.splitText(text);
        expect(result).toEqual(expectedChunks);
    });

    test('should return an empty array when input text is empty', async () => {
        const text = '';
        const expectedChunks: string[] = [];

        const result = await splitter.splitText(text);
        expect(result).toEqual(expectedChunks);
    });

    test('should return the entire text as a single chunk when there are no headers', async () => {
        const text = `
This is a novel without any markdown headers.
It should be treated as a single chunk.
        `;
        const expectedChunks = [
            'This is a novel without any markdown headers.\nIt should be treated as a single chunk.',
        ];

        const result = await splitter.splitText(text);
        expect(result).toEqual(expectedChunks);
    });

    test('should handle headers with special characters', async () => {
        const text = `
# Chapter 1: The Beginning!
Content for chapter 1.

## Chapter 2: What's Next?
Content for chapter 2.

### Chapter 3: The Finale.
Content for chapter 3.
        `;
        const expectedChunks = [
            'Content for chapter 1.',
            'Content for chapter 2.',
            'Content for chapter 3.',
        ];

        const result = await splitter.splitText(text);
        expect(result).toEqual(expectedChunks);
    });

    test('should handle multiple consecutive headers without content', async () => {
        const text = `
# Chapter 1

## Chapter 2

### Chapter 3
Content for chapter 3.
        `;
        const expectedChunks = [
            '',
            '',
            'Content for chapter 3.',
        ];

        const result = await splitter.splitText(text);
        expect(result).toEqual(expectedChunks);
    });

    test('should handle headers with irregular spacing', async () => {
        const text = `
#    Chapter 1
Content for chapter 1.

##\tChapter 2
Content for chapter 2.
        `;
        const expectedChunks = [
            'Content for chapter 1.',
            'Content for chapter 2.',
        ];

        const result = await splitter.splitText(text);
        expect(result).toEqual(expectedChunks);
    });

    test('should handle large text input efficiently', async () => {
        const numberOfChapters = 100;
        let text = '';
        const expectedChunks: string[] = [];

        for (let i = 1; i <= numberOfChapters; i++) {
            text += `# Chapter ${i}\nContent for chapter ${i}.\n\n`;
            expectedChunks.push(`Content for chapter ${i}.`);
        }

        const result = await splitter.splitText(text);
        expect(result).toEqual(expectedChunks);
    });

    test('should split multiple documents into chapters', async () => {
        const docs: Document[] = [
            new Document({ pageContent: `
# Doc1 Chapter 1
Content of Doc1 Chapter 1.

# Doc1 Chapter 2
Content of Doc1 Chapter 2.
            ` }),
            new Document({ pageContent: `
# Doc2 Chapter 1
Content of Doc2 Chapter 1.

# Doc2 Chapter 2
Content of Doc2 Chapter 2.
            ` }),
        ];

        const expectedChunks: Document[] = [
            new Document({ pageContent: 'Content of Doc1 Chapter 1.' }),
            new Document({ pageContent: 'Content of Doc1 Chapter 2.' }),
            new Document({ pageContent: 'Content of Doc2 Chapter 1.' }),
            new Document({ pageContent: 'Content of Doc2 Chapter 2.' }),
        ];

        const result = await splitter.splitDocuments(docs);
        expect(result).toEqual(expectedChunks);
    });

    test('should handle documents without headers as single chunks', async () => {
        const docs: Document[] = [
            new Document({ pageContent: 'This document has no headers.' }),
            new Document({ pageContent: 'Another headerless document.' }),
        ];

        const expectedChunks: Document[] = [
            new Document({ pageContent: 'This document has no headers.' }),
            new Document({ pageContent: 'Another headerless document.' }),
        ];

        const result = await splitter.splitDocuments(docs);
        expect(result).toEqual(expectedChunks);
    });

    test('should return an empty array when input documents array is empty', async () => {
        const docs: Document[] = [];
        const expectedChunks: Document[] = [];

        const result = await splitter.splitDocuments(docs);
        expect(result).toEqual(expectedChunks);
    });
});
