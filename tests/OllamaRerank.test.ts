import { describe, it, expect, beforeEach } from "bun:test";
import { DocumentInterface } from "@langchain/core/documents";
import { OllamaRerank, OllamaRerankArgs } from "../lib/OllamaRerank";

describe("OllamaRerank", () => {
    let reranker: OllamaRerank;
    const documents: DocumentInterface[] = [
        { pageContent: "Document 1", metadata: {} },
        { pageContent: "Document 2", metadata: {} },
    ];

    beforeEach(() => {
        reranker = new OllamaRerank({ model: "test-model" });
    });

    it("should initialize with default values", () => {
        expect(reranker.model).toBe("test-model");
        expect(reranker.topN).toBe(3);
    });

    it("should compress documents", async () => {
        const compressedDocs = await reranker.compressDocuments(documents, "query");
        expect(compressedDocs).toHaveLength(2);
        expect(compressedDocs[0].metadata.relevanceScore).toBe(0.9);
        expect(compressedDocs[1].metadata.relevanceScore).toBe(0.8);
    });

    it("should rerank documents", async () => {
        const results = await reranker.rerank(documents, "query");
        expect(results).toHaveLength(2);
        expect(results[0].relevanceScore).toBe(0.9);
        expect(results[1].relevanceScore).toBe(0.8);
    });
});
