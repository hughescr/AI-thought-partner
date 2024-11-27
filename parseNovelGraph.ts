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
import neo4j, { Driver } from 'neo4j-driver';
import { tool } from '@langchain/core/tools';
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

const EntitiesSchema = z.array(EntitySchema).describe('List of entities');
const RelationshipsSchema = z.array(RelationshipSchema).describe('List of relationships');

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
 * Generate a plain-text string representation of an entity.
 * @param {Entity} entity - The entity to represent as a string.
 * @returns {string} - The trimmed string representation of the entity.
 */
function entityToStringRepresentation(entity: Entity): string {
    const description = `
        Name: ${entity.name}
        Type: ${entity.type}
        Description: ${entity.description}
        Aliases: ${entity.aliases.join(', ')}
    `;
    return _.trim(description);
}

/**
 * Generate a plain-text description of an entity and calculate its embedding.
 * @param {Entity} entity - The entity to describe and calculate the embedding for.
 * @returns {Promise<number[]>} - The embedding of the entity description.
 */
async function generateEntityEmbedding(entity: Entity): Promise<number[]> {
    // Get the trimmed string representation of the entity
    const trimmedDescription = entityToStringRepresentation(entity);

    // Calculate the embedding of the trimmed description using the embeddings object
    return await embeddings.embedQuery(trimmedDescription);
}

/**
 * Retrieve matching documents for an entity and map them to ExtractWithContext objects.
 * @param {Entity} entity - The entity to search for in the vectorstore.
 * @returns {Promise<ExtractWithContext[]>} - An array of ExtractWithContext objects.
 */
async function getExtractsForEntity(entity: Entity): Promise<ExtractWithContext[]> {
    // Get the string representation of the entity
    const entityString = entityToStringRepresentation(entity);

    // Query the extractRetriever to get matching documents
    const documents = await extractRetriever.invoke(entityString);

    // Map the documents to ExtractWithContext objects
    return _.map(documents, doc => new ExtractWithContext(doc.pageContent, doc.metadata.context));
}

/**
 * Search the Neo4j store using gds.similarity.cosine to find entities matching the embedding of the given entity.
 * @param {Entity} entity - The entity to search for similar entities in the Neo4j store.
 * @returns {Promise<SimilarEntityResult[]>} - An array of found entities and their similarity scores.
 */
class SimilarEntityResult {
    entity: Entity;
    score: number;

    constructor(entity: Entity, score: number) {
        this.entity = entity;
        this.score = score;
    }
}

async function findSimilarEntitiesInNeo4j(entity: Entity, limit = 3): Promise<SimilarEntityResult[]> {
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
            LIMIT $limit
        `, { entityEmbedding, limit });

        // Process and return the results
        return _.map(result.records, record => new SimilarEntityResult(
            new Entity(
                record.get('e').properties.name,
                record.get('e').properties.type,
                record.get('e').properties.description,
                record.get('e').properties.aliases
            ),
            record.get('score')
        ));
    } finally {
        await session.close();
    }
}

/**
 * Update an existing entity in the Neo4j database with information from a replacement entity.
 * The match is based on name, type, and description.
 * @param {Entity} existingEntity - The existing entity to be updated.
 * @param {Entity} replacementEntity - The replacement entity with updated information.
 * @returns {Promise<void>} - A promise that resolves when the update is complete.
 */
async function updateEntityInNeo4j(existingEntity: Entity, replacementEntity: Entity): Promise<void> {
    const session = driver.session();

    try {
        // Calculate the embedding for the replacement entity
        const replacementEntityEmbedding = await generateEntityEmbedding(replacementEntity);

        // Cypher query to update the existing entity with the replacement entity's properties, including the embedding
        await session.run(`
            MATCH (e:Entity {name: $existingName, type: $existingType, description: $existingDescription})
            SET e.name = $replacementName,
                e.type = $replacementType,
                e.description = $replacementDescription,
                e.aliases = $replacementAliases,
                e.embedding = $replacementEmbedding
        `, {
            existingName: existingEntity.name,
            existingType: existingEntity.type,
            existingDescription: existingEntity.description,
            replacementName: replacementEntity.name,
            replacementType: replacementEntity.type,
            replacementDescription: replacementEntity.description,
            replacementAliases: replacementEntity.aliases,
            replacementEmbedding: replacementEntityEmbedding,
        });
    } finally {
        await session.close();
    }
}

/**
 * Insert a new entity into the Neo4j database.
 * @param {Entity} entity - The entity to be inserted.
 * @returns {Promise<void>} - A promise that resolves when the insertion is complete.
 */
async function insertEntityIntoNeo4j(entity: Entity): Promise<void> {
    const session = driver.session();

    try {
        // Calculate the embedding for the entity
        const entityEmbedding = await generateEntityEmbedding(entity);

        // Cypher query to create a new entity node with the given properties, including the embedding
        await session.run(`
            CREATE (e:Entity {
                name: $name,
                type: $type,
                description: $description,
                aliases: $aliases,
                embedding: $embedding
            })
        `, {
            name: entity.name,
            type: entity.type,
            description: entity.description,
            aliases: entity.aliases,
            embedding: entityEmbedding
        });
    } finally {
        await session.close();
    }
}

/**
 * Upsert an array of relationships into the Neo4j database.
 * Assumes that the entities involved in the relationships are already present in the database.
 * @param {Relationship[]} relationships - The array of relationships to upsert.
 * @returns {Promise<void>} - A promise that resolves when the upsert is complete.
 */
async function upsertRelationshipsIntoNeo4j(relationships: Relationship[]): Promise<void> {
    const session = driver.session();

    try {
        for(const relationship of relationships) {
            // Cypher query to merge the relationship if it doesn't already exist
            await session.run(`
                MATCH (source:Entity {name: $sourceName})
                MATCH (target:Entity {name: $targetName})
                MERGE (source)-[r:RELATIONSHIP {type: $type}]->(target)
                ON CREATE SET r.description = $description
            `, {
                sourceName: relationship.source,
                targetName: relationship.target,
                type: relationship.type,
                description: relationship.description || null
            });
        }
    } finally {
        await session.close();
    }
}

// END OF HELPER FUNCTIONS

// CHAINS

const entitiesLLM = fastDumbLLM.withStructuredOutput(EntitiesSchema);
const entitiesExtractionPrompt = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(`
You are an expert in text analysis. Your task is to extract entities from the given text extract.
You will receive a text extract in the user prompt.
Identify entities such as people, locations, organizations, themes, concepts, vehicles, and objects.
For each entity, provide a brief description and categorize it into one of the following types: Person, Location, Organization, Theme, Concept, Vehicle, Object.`
    ),
    HumanMessagePromptTemplate.fromTemplate(`Here is the text extract:
{extract}`),
]);
const entitiesExtractionChain = entitiesExtractionPrompt.pipe(entitiesLLM);

const relationshipsLLM = fastDumbLLM.withStructuredOutput(RelationshipsSchema);
const relationshipExtractionPrompt = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(`
You are an expert in relationship extraction. Your task is to identify relationships among the given entities within the provided text extract.
You will receive a text extract, and a list of entities. Identify any relationships between these entities, specifying the source, target, type, and an optional description for each relationship.`
    ),
    HumanMessagePromptTemplate.fromTemplate(`
Here is the text extract:
{extract}

Here is the list of entities:
{entities}`),
]);
const relationshipExtractionChain = relationshipExtractionPrompt.pipe(relationshipsLLM);

// Create a new structured output LLM for a single EntitySchema
const entityRefinementLLM = fastDumbLLM.withStructuredOutput(EntitySchema);
// Define the prompt for refining an entity
const entityRefinementPrompt = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(`
You are an expert in entity refinement. Your task is to refine a proposed entity using additional context.
You will receive a proposed entity, the original extract from which it was extracted, and supplemental extracts for context.
Your goal is to improve the entity's data, particularly its list of aliases and the name by which it is most commonly known.
Use the supplemental extracts to refine the entity's details and ensure the most accurate and complete representation.`
    ),
    HumanMessagePromptTemplate.fromTemplate(`
Proposed Entity:
{proposedEntity}

Original Extract:
{originalExtract}

Supplemental extracts:
{additionalExtracts}`
    )
]);
const entityRefinementChain = entityRefinementPrompt.pipe(entityRefinementLLM);

const updateEntityInNeo4jTool = tool(
    async ({ existingEntity, refinedEntity }: { existingEntity: Entity, refinedEntity: Entity }) => updateEntityInNeo4j(existingEntity, refinedEntity),
    {
        name: 'updateEntityInNeo4j',
        description: 'Update an existing entity in the Neo4j database with new information.',
        schema: z.object({
            existingEntity: EntitySchema,
            refinedEntity: EntitySchema
        })
    }
);

const insertEntityIntoNeo4jTool = tool(
    async ({ entity }: { entity: Entity }) => insertEntityIntoNeo4j(entity),
    {
        name: 'insertEntityIntoNeo4j',
        description: 'Insert a new entity into the Neo4j database.',
        schema: z.object({
            entity: EntitySchema
        })
    }
);

const entityAssessmentLLMWithTools = fastDumbLLM.bindTools([updateEntityInNeo4jTool, insertEntityIntoNeo4jTool]);
const entityAssessmentPrompt = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(`
You are an expert in entity assessment and refinement. Your task is to determine if a proposed entity matches any of the similar entities provided.
If a match is found, refine the proposed entity by updating its name, aliases, description, and type as necessary, and call the updateEntityInNeo4j tool.
If no match is found, call the insertEntityIntoNeo4j tool to add the proposed entity to the database.`
    ),
    HumanMessagePromptTemplate.fromTemplate(`
Here is the proposed entity:
{proposedEntity}

Here are the similar entities:
{similarEntities}`
    )
]);
const entityAssessmentChain = entityAssessmentPrompt.pipe(entityAssessmentLLMWithTools);

// END OF CHAINS

// WORKFLOW STAGE FUNCTIONS

async function extractEntities(state: { novelChunk: string }): Promise<{ entities: Entity[] }> {
    const entitiesResult = await entitiesExtractionChain.invoke({ extract: state.novelChunk });
    return { entities: entitiesResult };
}

/**
 * Refine entities using additional context from extracts.
 * @param {Object} state - The current state containing novel chunk and entities.
 * @returns {Promise<Object>} - The updated state with refined entities.
 */
async function refineEntitiesWithContext(state: { novelChunk: string, entities: Entity[] }): Promise<{ refinedEntities: Entity[] }> {
    const refinedEntities = await Promise.all(_.map(state.entities, async (entity) => {
        // Retrieve additional extracts for the entity
        const additionalExtracts = await getExtractsForEntity(entity);

        // Use entityRefinementChain to refine the entity with additional context
        const refinementResult = await entityRefinementChain.invoke({
            proposedEntity: entity,
            originalExtract: state.novelChunk,
            additionalExtracts: additionalExtracts
        });

        return refinementResult;
    }));

    return { refinedEntities };
}

// END OF WORKFLOW STAGE FUNCTIONS

// AGENT WORKFLOW

const ERExtractionAnnotation = Annotation.Root({
    novelMetadata: Annotation<NovelMetadata>,
    novelChunk: Annotation<string>,
    entities: Annotation<Entity[]>,
});

const workflow = new StateGraph(ERExtractionAnnotation)
    .addNode('Setup Metadata', setupMetadata)
    .addNode('Extract Entities', extractEntities)
    .addNode('Refine Entities with Context', refineEntitiesWithContext) // Add this line
    .addEdge(START, 'Setup Metadata')
    .addEdge('Setup Metadata', 'Extract Entities')
    .addEdge('Extract Entities', 'Refine Entities with Context') // Add this line
    .addEdge('Refine Entities with Context', END); // Add this line

const app = workflow.compile();

// END OF AGENT WORKFLOW

// Now run:
const initialState = {
    novelChunk: novelChunks[0],
};

for await (const output of await app.stream({ streamMode: 'values', recursionLimit: 50, initialState })) {
    logger.info(output);
}
