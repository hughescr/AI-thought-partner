import { Document, type DocumentInterface } from '@langchain/core/documents';
import { VectorStore } from '@langchain/core/vectorstores';
import { FaissStore } from '@langchain/community/vectorstores/faiss';
import type { Embeddings } from '@langchain/core/embeddings';
import { NovelDocument } from './NovelDocumentStore';
import { ChapterDocumentStore } from './ChapterDocumentStore';
import _ from 'lodash';

export class NovelFaissStore extends VectorStore {
    private store: FaissStore;
    private novel: NovelDocument;
    private chapterStore: ChapterDocumentStore;
    // This method must return a literal string for the VectorStore base class to work correctly
    // eslint-disable-next-line lodash/prefer-constant -- We need to return a literal directly to avoid runtime errors
    _vectorstoreType(): string { return 'novel_faiss'; }

    private constructor(store: FaissStore, novel: NovelDocument, chapterStore: ChapterDocumentStore) {
        super(store.embeddings, {});
        this.store = store;
        this.novel = novel;
        this.chapterStore = chapterStore;
    }

    public static async load(
        storePath: string,
        embeddings: Embeddings,
        novel: NovelDocument,
        chapterStore: ChapterDocumentStore
    ): Promise<NovelFaissStore> {
        let loadedStore: FaissStore;
        try {
            loadedStore = await FaissStore.load(storePath, embeddings);
        } catch(err: unknown) {
            const errorMessage = _.isError(err) ? err.message : String(err);
            throw new Error(`No FAISS store exists at path ${storePath} for novel ${novel.metadata.novelID} - ${errorMessage}`);
        }
        return new NovelFaissStore(loadedStore, novel, chapterStore);
    }

    async similaritySearchVectorWithScore(query: number[], k: number): Promise<[Document, number][]> {
    // Perform the regular vector search.
        const baseResults = await this.store.similaritySearchVectorWithScore(query, k);
        const results: [Document, number][] = [];
        const seenChapters = new Set<number>();
        for(const [doc, score] of baseResults) {
            if(doc.metadata?.chapter) {
                if(!seenChapters.has(doc.metadata.chapter)) {
                    // Fetch the full chapter document using the stored chapterStore.
                    const chapterDoc = await this.chapterStore.getChapter(this.novel, doc.metadata.chapter);
                    if(chapterDoc) {
                        results.push([chapterDoc, score]);
                        seenChapters.add(doc.metadata.chapter);
                    }
                }
            } else {
                throw new Error(`Missing chapter metadata on document: ${JSON.stringify(doc)}`);
            }
        }
        return results;
    }

    async addDocuments(_documents: DocumentInterface[], _options?: Record<string, unknown>): Promise<string[] | void> {
        throw new Error('Method not implemented.');
    }

    async addVectors(_vectors: number[][], _documents: DocumentInterface[], _options?: Record<string, unknown>): Promise<string[] | void> {
        throw new Error('Method not implemented.');
    }
}
