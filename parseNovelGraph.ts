import {
    cachedJinaV2BaseENEmbeddings as embeddings,
    qwen25_32bLLM as slowSmartLLM,
    nemo_12bLLM as fastDumbLLM
} from './lib/LLMs.ts';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { SemanticTextSplitter } from './lib/SemanticTextSplitter.ts';
import { FaissStoreWithMMR } from './lib/FAISSStoreWithMMR.ts';
import { END, START, StateGraph, Annotation } from '@langchain/langgraph';
import { tool } from '@langchain/core/tools';
import { ToolNode } from '@langchain/langgraph/prebuilt';
import neo4j, { Driver, Session } from 'neo4j-driver';
import z from 'zod';
import { logger } from '@hughescr/logger';
import _ from 'lodash';
import chalk from 'chalk';

if(process.versions.bun === undefined) {
    logger.info(chalk.greenBright('Running under Node, setting global dispatcher'));
    const { setGlobalDispatcher, Agent } = await import('undici');
    setGlobalDispatcher(new Agent({ headersTimeout: 0, bodyTimeout: 0 })); // ensure we wait for long ollama runs
} else {
    logger.warn(chalk.yellowBright('Running under Bun, not setting global dispatcher so LLMs might timeout'));
}

const neo4jURL = process.env.NEO4J_URI || '';
const neo4jUsername = process.env.NEO4J_USER || '';
const neo4jPassword = process.env.NEO4J_PASSWORD || '';

const driver: Driver = neo4j.driver(
    neo4jURL,
    neo4j.auth.basic(neo4jUsername, neo4jPassword)
);

const splitter = new SemanticTextSplitter({
    showProgress: false,
    initialChunkSize: 32, // Tokens!
    chunkSize: 2048, // Tokens!
    embeddings: embeddings, // Use fast embeddings for decent semantic splits
    embeddingBatchSize: 128,
});

const book = 'Christmas Town beta';
const storeDirectory = `novels/${book}`;
const loader: TextLoader = new TextLoader(`novels/${book}.md`);
const novelText = await loader.load();

interface NovelMetadata {
    title: string
    author: string
    today: string
    genre: string
}
/**
 * Call the retriever to find matching documents
 * @param {GraphState} state - The current state of the agent, including the query.
 * @returns {Promise<GraphState>} - The updated state with the documents added.
 */
async function setupMetadata(): Promise<{ novelMetadata: NovelMetadata }> {
    logger.debug('---METADATA---');

    return {
        novelMetadata: {
            title: 'Christmas Town',
            author: 'Erica S. Hughes',
            today: new Date().toISOString(),
            genre: 'Literary Fiction/Young Adult',
        },
    };
}

const vectorStore = await FaissStoreWithMMR.load(
    storeDirectory,
    embeddings
);

// Define the Zod schema for structured output
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

const structuredLlm = slowSmartLLM.withStructuredOutput(OutputSchema);

const ERExtractionAnnotation = Annotation.Root({
    novelMetadata: Annotation<NovelMetadata>,
});

async function extractEntitiesAndRelationships(chunk: string) {
    const result = await structuredLlm.call({
        input: chunk,
    });

    const { entities, relationships } = result;

    const session: Session = driver.session();
    try {
        for (const entity of entities) {
            await session.run(
                'MERGE (e:Entity {name: $name, type: $type, description: $description})',
                entity
            );
        }

        for (const relationship of relationships) {
            await session.run(
                `MATCH (a:Entity {name: $source}), (b:Entity {name: $target})
                 MERGE (a)-[r:RELATIONSHIP {type: $type, description: $description}]->(b)`,
                relationship
            );
        }
    } finally {
        await session.close();
    }
}

const workflow = new StateGraph(ERExtractionAnnotation)
    .addNode('processChunks', async (state) => {
        const chunks = await splitter.split(novelText);
        for (const chunk of chunks) {
            await extractEntitiesAndRelationships(chunk);
        }
        return state;
    })
    .addNode('setupMetadata', setupMetadata)
    .addEdge(START, 'setupMetadata')
    .addEdge('setupMetadata', 'processChunks')
    .addEdge('processChunks', END);

const app = workflow.compile();

for await (const output of await app.stream({ streamMode: 'values', recursionLimit: 50 })) {
    logger.info(output);
}
