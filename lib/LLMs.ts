import { ChatOllama, OllamaEmbeddings } from '@langchain/ollama';
import { CacheBackedEmbeddings } from 'langchain/embeddings/cache_backed';
import { InMemoryStore } from 'langchain/storage/in_memory';
import { OllamaRerank } from './OllamaRerank';

const commonOptions = { temperature: 1, seed: 19740822, keepAlive: '15m' };
const commonOptions32k = { ...commonOptions, numCtx: 32 * 1024 };
const commonOptions64k = { ...commonOptions, numCtx: 64 * 1024 };

// Embeddings
export const nomicEmbeddings = new OllamaEmbeddings({
    model: 'nomic-embed-text',
    requestOptions: { numCtx: 2048 },
});

export const bgeM3Embeddings = new OllamaEmbeddings({
    model: 'bge-m3',
    requestOptions: { numCtx: 8192 },
});

export const cachedNomicEmbeddings = CacheBackedEmbeddings.fromBytesStore(
    nomicEmbeddings,
    new InMemoryStore(),
    {
        namespace: coreEmbeddings.model,
    }
);

// LLMs
export const qwen25_1_5bLLM = new ChatOllama({ model: 'qwen2.5:1.5b-instruct-fp16', ...commonOptions32k });
export const llama32_3bLLM = new ChatOllama({ model: 'llama3.2:3b-instruct-fp16', ...commonOptions32k });
export const phi35_4bLLM = new ChatOllama({ model: 'phi3.5:3.8b-mini-instruct-fp16', ...commonOptions32k });
export const llama31_8bLLM = new ChatOllama({ model: 'llama3.1:8b-instruct-q8_0', ...commonOptions64k });
export const nemo_12bLLM = new ChatOllama({ model: 'mistral-nemo:12b-instruct-2407-q8_0', ...commonOptions32k });
export const qwen25_14bLLM = new ChatOllama({ model: 'qwen2.5:14b-instruct-q8_0', ...commonOptions32k });
export const phi3_14bLLM = new ChatOllama({ model: 'phi3:14b-medium-128k-instruct-q8_0', ...commonOptions32k });
export const qwen25_32bLLM = new ChatOllama({ model: 'qwen2.5:32b-instruct-q8_0', ...commonOptions32k });
export const llama31_70bLLM = new ChatOllama({ model: 'llama3.1:70b-instruct-q8_0', ...commonOptions32k });
export const bespokeMinicheckLLM = new ChatOllama({ model: 'bespoke-minicheck:7b-q8_0', ...commonOptions32k });

// Rerankers
export const jinaV1TinyENReranker = new OllamaRerank({ model: 'jina-reranker-v1-tiny-en:bf16', topN: 10 });
export const bgeV2M3Reranker = new OllamaRerank({ model: 'bge-reranker-v2-m3:bf16', topN: 5 });
export const cachedBgeM3Embeddings = CacheBackedEmbeddings.fromBytesStore(
    bgeM3Embeddings,
    new InMemoryStore(),
    {
        namespace: bgeM3Embeddings.model,
    }
);
