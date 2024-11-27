import {
    cachedJinaV2BaseENEmbeddings as embeddings,
    qwen25_32bLLM as structureableLLM
} from './lib/LLMs.ts';
import { CacheBackedEmbeddings } from 'langchain/embeddings/cache_backed';
import { ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate } from '@langchain/core/prompts';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import neo4j, { Driver, Session } from 'neo4j-driver';
import z from 'zod';
import _ from 'lodash';
import chalk from 'chalk';
import cliProgress from 'cli-progress';
import { logger } from '@hughescr/logger';

const neo4jURL = process.env.NEO4J_URI || '';
const neo4jUsername = process.env.NEO4J_USER || '';
const neo4jPassword = process.env.NEO4J_PASSWORD || '';

const book = 'Christmas Town beta';
const loader: TextLoader = new TextLoader(`novels/${book}.md`);
const novelText = await loader.load();

const EntitySchema = z.object({
    name: z.string().describe('The name of the entity'),
    type: z.enum(['Person', 'Location', 'Organization', 'Theme', 'Concept', 'Vehicle', 'Object']).describe('The type of the entity'),
    description: z.string().describe('Brief description or distinguishing details'),
});

const RelationshipSchema = z.object({
    source: z.string().describe('The name of the source entity'),
    target: z.string().describe('The name of the target entity'),
    type: z.string().describe('The type of the relationship'),
    description: z.string().optional().describe('Optional description'),
});

const OutputSchema = z.object({
    entities: z.array(EntitySchema).describe('List of extracted entities'),
    relationships: z.array(RelationshipSchema).describe('List of extracted relationships'),
}).describe('Extracted entities and relationships from the text.');

const structuredLlm = structureableLLM.withStructuredOutput(OutputSchema);

const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 8 * 1024,
    keepSeparator: true,
    separators: ['##'],
});

