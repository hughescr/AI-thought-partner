import { ChatOllama, OllamaEmbeddings } from '@langchain/ollama';
import { CacheBackedEmbeddings } from 'langchain/embeddings/cache_backed';
import { InMemoryStore } from 'langchain/storage/in_memory';
import { OllamaRerank } from './OllamaRerank.ts';

const commonOptions = { temperature: 1, seed: 19740822, keepAlive: '15m' };
const commonOptions32k = { ...commonOptions, numCtx: 32 * 1024 };
const commonOptions64k = { ...commonOptions, numCtx: 64 * 1024 };

// Embeddings

const makeCachedEmbeddings = (model: OllamaEmbeddings) => CacheBackedEmbeddings.fromBytesStore(
    model,
    new InMemoryStore(),
    {
        namespace: model.model,
    }
);

// Apache License 2.0
export const nomicEmbeddings = new OllamaEmbeddings({ model: 'nomic-embed-text', requestOptions: { numCtx: 2048 } });
export const cachedNomicEmbeddings = makeCachedEmbeddings(nomicEmbeddings);
export const jinaV2SmallENEmbeddings = new OllamaEmbeddings({ model: 'jina/jina-embeddings-v2-small-en', requestOptions: { numCtx: 8192 } });
export const cachedJinaV2SmallENEmbeddings = makeCachedEmbeddings(jinaV2SmallENEmbeddings);
export const jinaV2BaseENEmbeddings = new OllamaEmbeddings({ model: 'jina/jina-embeddings-v2-base-en', requestOptions: { numCtx: 8192 } });
export const cachedJinaV2BaseENEmbeddings = makeCachedEmbeddings(jinaV2BaseENEmbeddings);

// MIT License
export const bgeM3Embeddings = new OllamaEmbeddings({ model: 'bge-m3', requestOptions: { numCtx: 8192 } });
export const cachedBgeM3Embeddings = makeCachedEmbeddings(bgeM3Embeddings);

// LLMs

// Apache License 2.0
export const qwen25_1_5bLLM = new ChatOllama({ model: 'qwen2.5:1.5b-instruct-fp16', ...commonOptions32k });
export const qwen25_14bLLM = new ChatOllama({ model: 'qwen2.5:14b-instruct-q8_0', ...commonOptions32k });
export const qwen25_32bLLM = new ChatOllama({ model: 'qwen2.5:32b-instruct-q8_0', ...commonOptions32k });
export const nemo_12bLLM = new ChatOllama({ model: 'mistral-nemo:12b-instruct-2407-q8_0', ...commonOptions32k });

// MIT License
export const phi35_4bLLM = new ChatOllama({ model: 'phi3.5:3.8b-mini-instruct-fp16', ...commonOptions32k });
export const phi3_14bLLM = new ChatOllama({ model: 'phi3:14b-medium-128k-instruct-q8_0', ...commonOptions32k });

// Llama Community License
export const llama32_3bLLM = new ChatOllama({ model: 'llama3.2:3b-instruct-fp16', ...commonOptions32k });
export const llama31_8bLLM = new ChatOllama({ model: 'llama3.1:8b-instruct-q8_0', ...commonOptions64k });
export const llama31_70bLLM = new ChatOllama({ model: 'llama3.1:70b-instruct-q8_0', ...commonOptions32k });

// CC-Attribution-Non-Commercial
export const bespokeMinicheckLLM = new ChatOllama({ model: 'bespoke-minicheck:7b-q8_0', ...commonOptions32k });

// Rerankers

// Apache License 2.0
export const jinaV1TinyENReranker = new OllamaRerank({ model: 'jina-reranker-v1-tiny-en:bf16', topN: 10 });
export const bgeV2M3Reranker = new OllamaRerank({ model: 'bge-reranker-v2-m3:bf16', topN: 5 });
