import { mock, fn } from "bun:test";

mock("ollama/browser", () => {
    return {
        Ollama: class {
            rerank = fn().mockImplementation(({ model }) => {
                if (model !== "test-model") {
                    throw new Error(`model "${model}" not found, try pulling it first`);
                }
                return Promise.resolve({
                    results: [
                        { document: 0, relevance_score: 0.9 },
                        { document: 1, relevance_score: 0.8 },
                    ],
                });
            });
        },
    };
});
