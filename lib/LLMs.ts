import { ChatBedrockConverse } from '@langchain/aws';
import { OpenAIEmbeddings } from '@langchain/openai';
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
export const snowflakeArctic2LMEmbeddings = new OpenAIEmbeddings({ configuration: { baseURL: 'http://localhost:1234/v1' }, model: 'text-embedding-snowflake-arctic-embed-l-v2.0' });
export const cachedSnowflakeArctic2LMEmbeddings = makeCachedEmbeddings(snowflakeArctic2LMEmbeddings);

// LLMs

// Apache License 2.0
// export const qwen25_1_5bLLM = new ChatOllama({ model: 'qwen2.5:1.5b-instruct-q6_k_l', ...commonOptions32k });
// export const qwen25_3bLLM = new ChatOllama({ model: 'qwen2.5:3b-instruct-q6_k_l', ...commonOptions32k });
// export const qwen25_14bLLM = new ChatOllama({ model: 'qwen2.5:14b-instruct-q6_k_l', ...commonOptions32k });
// export const qwen25_32bLLM = new ChatOllama({ model: 'qwen2.5:32b-instruct-q6_k_l', ...commonOptions32k });
// export const nemo_12bLLM = new ChatOllama({ model: 'mistral-nemo:12b-instruct-2407-q6_k_l', ...commonOptions32k });

// // MIT License
// export const phi35_4bLLM = new ChatOllama({ model: 'phi3.5:3.8b-mini-instruct-q6_k_l', ...commonOptions32k });
// export const phi3_14bLLM = new ChatOllama({ model: 'phi3:14b-medium-128k-instruct-q6_k_l', ...commonOptions32k });

// // Llama Community License
// export const llama32_3bLLM = new ChatOllama({ model: 'llama3.2:3b-instruct-q6_k_l', ...commonOptions32k });
// export const llama31_8bLLM = new ChatOllama({ model: 'llama3.1:8b-instruct-q6_k_l', ...commonOptions64k });
// export const llama33_70bLLM = new ChatOllama({ model: 'llama3.3:70b-instruct-q6_k_l', ...commonOptions64k });

// // CC-Attribution-Non-Commercial
// export const bespokeMinicheckLLM = new ChatOllama({ model: 'bespoke-minicheck:7b-q8_0', ...commonOptions32k });

// // Rerankers

// // Apache License 2.0
// export const jinaV1TinyENReranker = new OllamaRerank({ model: 'jina-reranker-v1-tiny-en:bf16', topN: 10 });
// export const jinaV1TurboENReranker = new OllamaRerank({ model: 'jina-reranker-v1-turbo-en:bf16', topN: 10 });
// export const bgeV2M3Reranker = new OllamaRerank({ model: 'bge-reranker-v2-m3:bf16', topN: 5 });
