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

// LLMs

// Apache License 2.0
// export const qwen25_1_5bLLM = new ChatOllama({ model: 'qwen2.5:1.5b-instruct-q6_k_l', ...commonOptions32k });
// export const qwen25_3bLLM = new ChatOllama({ model: 'qwen2.5:3b-instruct-q6_k_l', ...commonOptions32k });
export const qwen25_7bLLM = new ChatOpenAI({ model: 'qwen2.5-7b-instruct-1m', configuration: { baseURL: 'http://localhost:1234/v1' } });
export const qwen25_14bLLM = new ChatOpenAI({ model: 'qwen2.5-14b-instruct-1m', configuration: { baseURL: 'http://localhost:1234/v1' } });
export const qwen25_32bLLM = new ChatOpenAI({ model: 'qwen2.5-32b-instruct', configuration: { baseURL: 'http://localhost:1234/v1' } });
export const nemo_12b2407LLM = new ChatOpenAI({ model: 'mistral-nemo-instruct-2407', configuration: { baseURL: 'http://localhost:1234/v1' } });
export const mistralSmall_24b2501LLM = new ChatOpenAI({ model: 'mistral-small-24b-instruct-2501', configuration: { baseURL: 'http://localhost:1234/v1' } });

// // MIT License
export const phi4_14bLLM = new ChatOpenAI({ model: 'phi-4', configuration: { baseURL: 'http://localhost:1234/v1' } });
// export const phi35_4bLLM = new ChatOllama({ model: 'phi3.5:3.8b-mini-instruct-q6_k_l', ...commonOptions32k });
// export const phi3_14bLLM = new ChatOllama({ model: 'phi3:14b-medium-128k-instruct-q6_k_l', ...commonOptions32k });
export const deepseekR1_Qwen1_5bLLM = new ChatOpenAI({ model: 'deepseek-r1-distill-qwen-1.5b', configuration: { baseURL: 'http://localhost:1234/v1' } });
export const deepseekR1_Qwen14bLLM = new ChatOpenAI({ model: 'deepseek-r1-distill-qwen-14b', configuration: { baseURL: 'http://localhost:1234/v1' } });
export const deepseekR1_Qwen32bLLM = new ChatOpenAI({ model: 'deepseek-r1-distill-qwen-32b-mlx', configuration: { baseURL: 'http://localhost:1234/v1' } });
export const deepseekR1_Llama70bLLM = new ChatOpenAI({ model: 'deepseek-r1-distill-llama-70b', configuration: { baseURL: 'http://localhost:1234/v1' } });

// // Llama Community License
// export const llama32_3bLLM = new ChatOllama({ model: 'llama3.2:3b-instruct-q6_k_l', ...commonOptions32k });
// export const llama31_8bLLM = new ChatOllama({ model: 'llama3.1:8b-instruct-q6_k_l', ...commonOptions64k });
export const llama33_70bLLM = new ChatOpenAI({ model: 'llama-3.3-70b-instruct', configuration: { baseURL: 'http://localhost:1234/v1' } });

// // CC-Attribution-Non-Commercial
// export const bespokeMinicheckLLM = new ChatOllama({ model: 'bespoke-minicheck:7b-q8_0', ...commonOptions32k });

// // Rerankers

// // Apache License 2.0
// export const jinaV2RerankBaseMultilingual = new LlamaCppRerank({ modelPath: '/Users/craig/.lmstudio/models/gpustack/jina-reranker-v2-base-multilingual-GGUF/jina-reranker-v2-base-multilingual-FP16.gguf', topN: 10 });
export const bgeV2M3Reranker = new LlamaCppRerank({ modelPath: '/Users/craig/.lmstudio/models/gpustack/bge-reranker-v2-m3-GGUF/bge-reranker-v2-m3-FP16.gguf', topN: 5 });
// export const snowflakeArctic2Reranker = new LlamaCppRerank({ modelPath: '/Users/craig/.lmstudio/models/limcheekin/snowflake-arctic-embed-l-v2.0-GGUF/snowflake-arctic-embed-l-v2.0.F16.gguf', topN: 5 });
// export const jinaV1TinyENReranker = new LlamaCppRerank({ modelPath: '/Users/craig/.lmstudio/models/mradermacher/jina-reranker-v1-tiny-en-GGUF/jina-reranker-v1-tiny-en.f16.gguf', topN: 10 });
// export const jinaV1TurboENReranker = new LlamaCppRerank({ modelPath: '/Users/craig/.lmstudio/models/mradermacher/jina-reranker-v1-turbo-en-GGUF/jina-reranker-v1-turbo-en.f16.gguf', topN: 10 });
