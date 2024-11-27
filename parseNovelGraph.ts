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

// NOVEL DATA
const book = 'Christmas Town beta';

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

const storeDirectory = `novels/${book}`;
const loader: TextLoader = new TextLoader(`novels/${book}.md`);
const novelText = await loader.load();
const novelChunks = await splitter.splitDocuments(novelText);
const vectorStore = await FaissStoreWithMMR.load(
    storeDirectory,
    embeddings
);
const extractRetriever = vectorStore.asRetriever({
    k: 10,
});
// END OF NOVEL DATA

// WORKFLOW STAGE FUNCTIONS

// END OF WORKFLOW STAGE FUNCTIONS

// SCHEMAS

const EntitySchema = z.object({
    name: z.string().describe('The name of the entity'),
    aliases: z.array(z.string()).describe('Known aliases for the entity, if any'),
    type: z.enum(['Person', 'Location', 'Organization', 'Theme', 'Concept', 'Vehicle', 'Object']).describe('The type of the entity'),
    description: z.string().describe('Brief description or distinguishing details'),
});

const RelationshipSchema = z.object({
    source: z.string().describe('The name of the source entity'),
    target: z.string().describe('The name of the target entity'),
    type: z.string().describe('The type of the relationship'),
    description: z.string().optional().describe('Optional description'),
});

const EntitiesAndRelationshipsSchema = z.object({
    entities: z.array(EntitySchema).describe('List of entities'),
    relationships: z.array(RelationshipSchema).describe('List of relationships'),
}).describe('Entities and relationships');

// END OF SCHEMAS

// CLASSES

class Entity {
    name: string;
    aliases: string[];
    type: 'Person' | 'Location' | 'Organization' | 'Theme' | 'Concept' | 'Vehicle' | 'Object';
    description: string;

    constructor(name: string, type: 'Person' | 'Location' | 'Organization' | 'Theme' | 'Concept' | 'Vehicle' | 'Object', description: string, aliases: string[] = []) {
        this.name = name;
        this.aliases = aliases;
        this.type = type;
        this.description = description;
    }
}

class Relationship {
    source: string;
    target: string;
    type: string;
    description?: string;

    constructor(source: string, target: string, type: string, description?: string) {
        this.source = source;
        this.target = target;
        this.type = type;
        this.description = description;
    }
}

class ExtractWithContext {
    extract: string;
    context: string;

    constructor(extract: string, context: string) {
        this.extract = extract;
        this.context = context;
    }
}

// END OF CLASSES

// HELPER FUNCTIONS

/**
 * Generate a plain-text description of an entity and calculate its embedding.
 * @param {Entity} entity - The entity to describe and calculate the embedding for.
 * @returns {Promise<number[]>} - The embedding of the entity description.
 */
async function generateEntityEmbedding(entity: Entity): Promise<number[]> {
    // Generate a plain-text description of the entity
    const description = `
        Name: ${entity.name}
        Type: ${entity.type}
        Description: ${entity.description}
        Aliases: ${entity.aliases.join(', ')}
    `;

    // Trim the description using lodash
    const trimmedDescription = _.trim(description);

    // Calculate the embedding of the trimmed description using the embeddings object
    return await embeddings.embedQuery(trimmedDescription);
}

/**
 * Search the Neo4j store using gds.similarity.cosine to find entities matching the embedding of the given entity.
 * @param {Entity} entity - The entity to search for similar entities in the Neo4j store.
 * @returns {Promise<{ entity: Entity, score: number }[]>} - An array of found entities and their similarity scores.
 */
async function findSimilarEntitiesInNeo4j(entity: Entity): Promise<{ entity: Entity, score: number }[]> {
    const session = driver.session();

    try {
        // Generate the embedding for the given entity
        const entityEmbedding = await generateEntityEmbedding(entity);

        // Cypher query to find similar entities using cosine similarity
        const result = await session.run(`
            MATCH (e:Entity)
            WHERE e.embedding IS NOT NULL
            RETURN e, gds.similarity.cosine(e.embedding, $entityEmbedding) AS score
            ORDER BY score DESC
        `, { entityEmbedding });

        // Process and return the results
        return _.map(result.records, record => ({
            entity: record.get('e').properties,
            score: record.get('score')
        }));
    } finally {
        await session.close();
    }
}

// END OF HELPER FUNCTIONS

// CHAINS

const entitiesAndRelationshipsLLM = fastDumbLLM.withStructuredOutput(EntitiesAndRelationshipsSchema);
const extractionPrompt = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(`
You are an expert in text analysis. Your task is to extract entities and relationships from the given text extract and its context.
You will receive a text extract and some context about it in the user prompt.
Identify entities such as people, locations, organizations, themes, concepts, vehicles, and objects.
For each entity, provide a brief description and categorize it into one of the following types: Person, Location, Organization, Theme, Concept, Vehicle, Object.
Also, identify relationships between these entities, specifying the type and a brief description of each relationship.`
    ),
    HumanMessagePromptTemplate.fromTemplate(`Here is the text extract:
{extract}

Here is some context about the extract:
{context}`),
]);
const extractionChain = extractionPrompt.pipe(entitiesAndRelationshipsLLM);

// Create a new structured output LLM for a single EntitySchema
const entityRefinementLLM = fastDumbLLM.withStructuredOutput(EntitySchema);
// Define the prompt for refining an entity
const entityRefinementPrompt = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(`
You are an expert in entity refinement. Your task is to refine a proposed entity using additional context.
You will receive a proposed entity, the original extract from which it was extracted, and additional extracts for context.
Your goal is to improve the entity's data, particularly its list of aliases and the name by which it is most commonly known.
Use the additional extracts to refine the entity's details and ensure the most accurate and complete representation.`
    ),
    HumanMessagePromptTemplate.fromTemplate(`
Here is the proposed entity:
{proposedEntity}

Here is the original extract and context:
Extract: {originalExtract}
Context: {originalContext}

Here are additional extracts for context:
{additionalExtracts}`
    )
]);
const entityRefinementChain = entityRefinementPrompt.pipe(entityRefinementLLM);

// END OF CHAINS

// AGENT WORKFLOW

const ERExtractionAnnotation = Annotation.Root({
    novelMetadata: Annotation<NovelMetadata>,
});

const workflow = new StateGraph(ERExtractionAnnotation)
    .addNode('Setup Metadata', setupMetadata)
    .addEdge(START, 'Setup Metadata')
    .addEdge('Setup Metadata', END);

const app = workflow.compile();

// END OF AGENT WORKFLOW

// Now run:
for await (const output of await app.stream({ streamMode: 'values', recursionLimit: 50 })) {
    logger.info(output);
}
