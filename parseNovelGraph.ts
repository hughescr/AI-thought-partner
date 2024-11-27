import {
    cachedJinaV2BaseENEmbeddings as embeddings,
    qwen25_32bLLM as slowSmartLLM,
    nemo_12bLLM as fastDumbLLM
} from './lib/LLMs.ts';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { SemanticTextSplitter } from './lib/SemanticTextSplitter.ts';
import { FaissStoreWithMMR } from './lib/FAISSStoreWithMMR.ts';
import { END, START, StateGraph, Annotation } from '@langchain/langgraph';
import { ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate } from '@langchain/core/prompts';
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
const novelChunks = await splitter.splitDocuments(novelText);

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
const extractRetriever = vectorStore.asRetriever({
    k: 10,
});

// Define the Zod schema for structured output
const EntitySchema = z.object({
    name: z.string().describe('The name of the entity'),
    aliases: z.array(z.string()).describe('Known aliases for the entity'),
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

const structuredLlm = fastDumbLLM.withStructuredOutput(OutputSchema);

const extractionPrompt = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(`
You are an expert in text analysis. Your task is to extract entities and relationships from the given text extract and its context.
You will receive a text extract and some context about it in the user prompt.
Identify entities such as people, locations, organizations, themes, concepts, vehicles, and objects.
For each entity, provide a brief description and categorize it into one of the following types: Person, Location, Organization, Theme, Concept, Vehicle, Object.
Also, identify relationships between these entities, specifying the type and a brief description of each relationship.`
    ),
    HumanMessagePromptTemplate.fromTemplate(`Here is the text extract:
{chunk}

Here is some context about the extract:
{context}`),
]);

const ERExtractionAnnotation = Annotation.Root({
    novelMetadata: Annotation<NovelMetadata>,
});

async function extractEntitiesAndRelationships(chunk: string) {
    const results = await extractRetriever
        .withConfig({ runName: 'FetchRelevantExtracts' })
        .invoke(chunk);

    const formattedResults = _.map(results, doc => ({
        chunk: doc.pageContent,
        context: doc.metadata.context,
    }));

    const session: Session = driver.session();
    try {
        for(const piece of formattedResults) {
            const { entities, relationships } = await extractionPrompt.pipe(structuredLlm).invoke(piece);

            // Use vectorStore to disambiguate entities
            for(const entity of entities) {
                const context = await extractRetriever
                    .withConfig({ runName: 'FetchRelevantExtracts' })
                    .invoke(entity.name);
                entity.description += ` Context: ${context}`;
            }

            for(const entity of entities) {
                await session.run(
                    `MERGE (e:Entity {name: $name})
                    ON CREATE SET e.aliases = $aliases, e.type = $type, e.description = $description
                    ON MATCH SET e.aliases = apoc.coll.union(e.aliases, $aliases)`,
                    {
                        name: entity.name,
                        aliases: entity.aliases,
                        type: entity.type,
                        description: entity.description
                    }
                );
            }

            for(const relationship of relationships) {
                await session.run(
                    `MATCH (a:Entity {name: $source}), (b:Entity {name: $target})
                    MERGE (a)-[r:RELATIONSHIP {type: $type, description: $description}]->(b)`,
                    relationship
                );
            }
        }
    } finally {
        await session.close();
    }
}

const refineEntityPrompt = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(`
You are an expert in entity refinement. Your task is to refine a proposed entity extracted from a novel using additional context from the novel.
You will receive a proposed entity, the original extract, its context, and additional extracts that might be relevant.
Use the additional extracts to refine the entity proposal, update the list of aliases, and ensure the entity's name is the most common form.`),
    HumanMessagePromptTemplate.fromTemplate(`
Original proposed entity: {entity}
Original extract: {extract}
Original context: {context}
Additional documents: {additionalDocuments}`),
]);

const structuredRefineLlm = slowSmartLLM.withStructuredOutput(EntitySchema);

async function refineEntity(entity, extract, context) {
    const query = `Name: ${entity.name}. Description: ${entity.description}. Known Aliases: ${entity.aliases.join(', ')}`;
    const additionalExtracts = await extractRetriever
        .withConfig({ runName: 'FetchRelevantExtracts' })
        .invoke(query);

    const refinedEntity = await refineEntityPrompt.pipe(structuredRefineLlm).invoke({
        entity: JSON.stringify(entity),
        extract,
        context,
        additionalDocuments,
    });

    return refinedEntity;
}

async function processNovelChunks() {
    const session: Session = driver.session();
    try {
        for(const chunk of novelChunks) {
            const { entities, relationships } = await extractEntitiesAndRelationships(chunk.pageContent);

            for(const entity of entities) {
                const refinedEntity = await refineEntity(entity, chunk.pageContent, chunk.metadata.context);

                await session.run(
                    `MERGE (e:Entity {name: $name})
                    ON CREATE SET e.aliases = $aliases, e.entityType = $entityType, e.description = $description
                    ON MATCH SET e.aliases = apoc.coll.union(e.aliases, $aliases)`,
                    {
                        name: refinedEntity.name,
                        aliases: refinedEntity.aliases,
                        entityType: refinedEntity.type,
                        description: refinedEntity.description
                    }
                );
            }

            for(const relationship of relationships) {
                await session.run(
                    `MATCH (a:Entity {name: $source}), (b:Entity {name: $target})
                    MERGE (a)-[r:RELATIONSHIP {type: $type, description: $description}]->(b)`,
                    relationship
                );
            }
        }
    } finally {
        await session.close();
    }
}

const workflow = new StateGraph(ERExtractionAnnotation)
    .addNode('setupMetadata', setupMetadata)
    .addNode('processNovelChunks', processNovelChunks)
    .addEdge(START, 'setupMetadata')
    .addEdge('setupMetadata', 'processNovelChunks')
    .addEdge('processNovelChunks', END);

const app = workflow.compile();

for await (const output of await app.stream({ streamMode: 'values', recursionLimit: 50 })) {
    logger.info(output);
}
