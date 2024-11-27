import {
    cachedJinaV2BaseENEmbeddings as embeddings,
    qwen25_32bLLM as slowSmartLLM,
    nemo_12bLLM as fastDumbLLM,
} from './lib/LLMs.ts';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import neo4j, { Driver, Session } from 'neo4j-driver';
import z from 'zod';

const neo4jURL = process.env.NEO4J_URI || '';
const neo4jUsername = process.env.NEO4J_USER || '';
const neo4jPassword = process.env.NEO4J_PASSWORD || '';

const driver: Driver = neo4j.driver(
    neo4jURL,
    neo4j.auth.basic(neo4jUsername, neo4jPassword)
);
const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 8 * 1024,
    keepSeparator: true,
    separators: ['##'],
});

const book = 'Christmas town beta';
const loader: TextLoader = new TextLoader(`novels/${book}.md`);
const novelText = await loader.load();
