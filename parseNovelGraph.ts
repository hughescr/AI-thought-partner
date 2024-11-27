import {
    cachedJinaV2BaseENEmbeddings as embeddings,
    qwen25_32bLLM as slowSmartLLM,
    nemo_12bLLM as fastDumbLLM
} from './lib/LLMs.ts';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { SemanticTextSplitter } from './lib/SemanticTextSplitter.ts';
import { FaissStoreWithMMR } from './lib/FAISSStoreWithMMR.ts';
import { END, START, StateGraph, Annotation } from '@langchain/langgraph';
import { ChatAnthropic } from '@langchain/anthropic';
import { tool } from '@langchain/core/tools';
import { z } from 'zod';
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

const vectorStore = await FaissStoreWithMMR.load(
    storeDirectory,
    embeddings
);

// Define a tool for entity extraction
const entityExtractionTool = tool(async ({ text }) => {
    // Placeholder for actual entity extraction logic
    // This should return entities and relationships found in the text
    return extractEntitiesAndRelationships(text);
}, {
    name: 'entityExtraction',
    description: 'Extract entities and relationships from text.',
    schema: z.object({
        text: z.string().describe('The text to analyze for entities and relationships.'),
    }),
});

const tools = [entityExtractionTool];
const toolNode = new ToolNode(tools);

const model = new ChatAnthropic({
    model: 'claude-3-5-sonnet-20240620',
    temperature: 0,
}).bindTools(tools);

async function extractEntitiesAndRelationships(text: string) {
    // Use vectorStore to help identify and disambiguate entities
    // Placeholder for actual logic
    return {
        entities: [
            { name: 'ACME Corp', type: 'Organization' },
            { name: 'Q2 2023', type: 'TimePeriod' }
        ],
        relationships: [
            { from: 'ACME Corp', to: 'Q2 2023', type: 'Reported' }
        ]
    };
}

async function storeInNeo4j(entities, relationships) {
    const session: Session = driver.session();
    try {
        for(const entity of entities) {
            await session.run(
                'MERGE (e:Entity {name: $name, type: $type})',
                { name: entity.name, type: entity.type }
            );
        }
        for(const relationship of relationships) {
            await session.run(
                'MATCH (a:Entity {name: $from}), (b:Entity {name: $to}) '
                + 'MERGE (a)-[:RELATIONSHIP {type: $type}]->(b)',
                { from: relationship.from, to: relationship.to, type: relationship.type }
            );
        }
    } finally {
        await session.close();
    }
}

const workflow = new StateGraph(Annotation.Root({
    messages: Annotation<BaseMessage[]>({
        reducer: (x, y) => x.concat(y),
    })
}))
    .addNode('agent', async (state) => {
        const messages = state.messages;
        const response = await model.invoke(messages);
        return { messages: [response] };
    })
    .addNode('tools', toolNode)
    .addEdge('__start__', 'agent')
    .addConditionalEdges('agent', (state) => {
        const messages = state.messages;
        const lastMessage = messages[messages.length - 1];
        if(lastMessage.tool_calls?.length) {
            return 'tools';
        }
        return '__end__';
    })
    .addEdge('tools', 'agent');

const checkpointer = new MemorySaver();
const app = workflow.compile({ checkpointer });

const chunks = splitter.split(novelText);
for(const chunk of chunks) {
    const finalState = await app.invoke(
        { messages: [new HumanMessage(chunk)] },
        { configurable: { thread_id: '42' } }
    );
    const { entities, relationships } = await extractEntitiesAndRelationships(chunk);
    await storeInNeo4j(entities, relationships);
}
