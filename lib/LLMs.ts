import { ChatBedrockConverse } from '@langchain/aws';
import { OpenAIEmbeddings, ChatOpenAI } from '@langchain/openai';
import { LlamaCppRerank } from './LlamaCppRerank';
import type { Embeddings } from '@langchain/core/embeddings';
import { CacheBackedEmbeddings } from 'langchain/embeddings/cache_backed';
import { InMemoryStore } from 'langchain/storage/in_memory';

// const commonOptions = { temperature: 0.2, seed: 19740822, keepAlive: '15m' };
// const commonOptions32k = { ...commonOptions, numCtx: 32 * 1024 };
// const commonOptions64k = { ...commonOptions, numCtx: 64 * 1024 };

// Embeddings

const makeCachedEmbeddings = (model: Embeddings) => CacheBackedEmbeddings.fromBytesStore(
    model,
    new InMemoryStore(),
    {
        namespace: 'embeddings',
    }
);

// AWS Bedrock
export const novaLiteLLM = new ChatBedrockConverse({
    model: 'us.amazon.nova-lite-v1:0',
    temperature: 1,
    topP: 1,
    additionalModelRequestFields: {
        inferenceConfig: {
            topK: 1,
        }
    },
    maxTokens: 2048,
    region: process.env.AWS_REGION,
    credentials: {
        accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
    },
    // verbose: true,
});
export const novaProLLM = new ChatBedrockConverse({
    model: 'us.amazon.nova-pro-v1:0',
    temperature: 1,
    topP: 1,
    additionalModelRequestFields: {
        inferenceConfig: {
            topK: 1,
        }
    },
    maxTokens: 2048,
    region: process.env.AWS_REGION,
    credentials: {
        accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
    },
    // verbose: true,
});
export const llama33bedrock_70bLLM = new ChatBedrockConverse({
    model: 'us.meta.llama3-3-70b-instruct-v1:0',
    temperature: 0,
    maxTokens: 2048,
    region: process.env.AWS_REGION,
    credentials: {
        accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
    },
    // verbose: true,
});

// Apache License 2.0
export const snowflakeArctic2Embeddings = new OpenAIEmbeddings({ model: 'text-embedding-snowflake-arctic-embed-l-v2.0', configuration: { baseURL: 'http://localhost:1234/v1' } });
export const cachedSnowflakeArctic2Embeddings = makeCachedEmbeddings(snowflakeArctic2Embeddings);
export const mxBAIEmbeddings = new OpenAIEmbeddings({ model: 'text-embedding-mxbai-embed-large-v1', configuration: { baseURL: 'http://localhost:1234/v1' } });
export const cachedMxBAIEmbeddings = makeCachedEmbeddings(mxBAIEmbeddings);
export const qwen3_06_Embeddings = new OpenAIEmbeddings({ model: 'qwen3-embedding-0.6b', configuration: { baseURL: 'http://localhost:1234/v1' } });
export const cachedQwen3_06_Embeddings = makeCachedEmbeddings(qwen3_06_Embeddings);

// LLMs

// Apache License 2.0
export const qwen3_0_6bLLM = new ChatOpenAI({ model: 'qwen3-0.6b', configuration: { baseURL: 'http://localhost:1234/v1' }, streaming: true });
export const qwen3_30bA3bLLM = new ChatOpenAI({ model: 'qwen3-30b-a3b-mlx', configuration: { baseURL: 'http://localhost:1234/v1' }, streaming: true });
export const deepseekR1_Qwen3_8bLLM = new ChatOpenAI({ model: 'deepseek/deepseek-r1-0528-qwen3-8b', configuration: { baseURL: 'http://localhost:1234/v1' }, streaming: true });
export const nemo_12b2407LLM = new ChatOpenAI({ model: 'mistral-nemo-instruct-2407', configuration: { baseURL: 'http://localhost:1234/v1' }, streaming: true });
export const mistralSmall_24b2503LLM = new ChatOpenAI({ model: 'mistral-small-3.1-24b-instruct-2503', configuration: { baseURL: 'http://localhost:1234/v1' }, streaming: true });

// // MIT License
export const deepseekR1_Qwen1_5bLLM = new ChatOpenAI({ model: 'deepseek-r1-distill-qwen-1.5b', configuration: { baseURL: 'http://localhost:1234/v1' }, streaming: true });
export const deepseekR1_Qwen32bLLM = new ChatOpenAI({ model: 'deepseek-r1-distill-qwen-32b-mlx', configuration: { baseURL: 'http://localhost:1234/v1' }, streaming: true });

// // Llama Community License
// export const llama32_3bLLM = new ChatOllama({ model: 'llama3.2:3b-instruct-q6_k_l', ...commonOptions32k });
// export const llama31_8bLLM = new ChatOllama({ model: 'llama3.1:8b-instruct-q6_k_l', ...commonOptions64k });
// export const llama33_70bLLM = new ChatOpenAI({ model: 'llama-3.3-70b-instruct', configuration: { baseURL: 'http://localhost:1234/v1' }, streaming: true });

// // CC-Attribution-Non-Commercial
// export const bespokeMinicheckLLM = new ChatOllama({ model: 'bespoke-minicheck:7b-q8_0', ...commonOptions32k });

// // Rerankers

// // Apache License 2.0
// export const jinaV2RerankBaseMultilingual = new LlamaCppRerank({ modelPath: '/Users/craig/.lmstudio/models/gpustack/jina-reranker-v2-base-multilingual-GGUF/jina-reranker-v2-base-multilingual-FP16.gguf', topN: 10 });
export const bgeV2M3Reranker = new LlamaCppRerank({ modelPath: '/Users/craig/.lmstudio/models/gpustack/bge-reranker-v2-m3-GGUF/bge-reranker-v2-m3-FP16.gguf', topN: 5 });
