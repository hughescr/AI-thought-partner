import {
    cachedJinaV2BaseENEmbeddings as embeddings,
    llama33_70bLLM as LLM
} from './lib/LLMs.ts';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { FaissStoreWithMMR } from './lib/FAISSStoreWithMMR.ts';
import { END, START, StateGraph, Annotation } from '@langchain/langgraph';
import { ChatPromptTemplate, HumanMessagePromptTemplate } from '@langchain/core/prompts';
import neo4j from 'neo4j-driver';
import { tool } from '@langchain/core/tools';
import { ToolNode } from '@langchain/langgraph/prebuilt';
import z from 'zod';
import { logger } from '@hughescr/logger';
import cliProgress from 'cli-progress';
import _ from 'lodash';
import chalk from 'chalk';
import { execa } from 'execa';

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

const driver = neo4j.driver(
    neo4jURL,
    neo4j.auth.basic(neo4jUsername, neo4jPassword)
);

const bars = new cliProgress.MultiBar({
    clearOnComplete: true,
    hideCursor: false,
    format: '{bar} {percentage}% | {duration_formatted} | ETA: {eta_formatted} | {value}/{total} | {name}',
}, cliProgress.Presets.shades_classic);

// Strategy per https://blog.getbind.co/2024/09/25/claude-contextual-retrieval-vs-rag-how-is-it-different/
// Break the source into large chunks (maybe 8k tokens semantically)
// Then, for each large chunk, break it into small chunks (maybe 256 tokens), and ask an LLM to describe the context of each small chunk (adding another 256 tokens)
// Then, concat the context and the extract, calculate encodings, and store in a vector store

class RecursiveCharacterTextSplitterSeparatorMod extends RecursiveCharacterTextSplitter {
    // Override the splitOnSeparator method to allow for keeping the separator attached to the earlier chunk not the later chunk
    // Without this, punctuation ends up on the wrong chunk... for example
    // Sentence 1. Sentence 2. ==> ['Sentence 1', '. Sentence 2', '.'] instead of ['Sentence 1.', 'Sentence 2.' ]
    // The former is clearly dumber than shit and will confuse the LLM with its weird leading periods and no end to the sentence, etc.
    splitOnSeparator(text: string, separator: string): string[] {
        let splits: string[] = [];
        if(separator) {
            if(this.keepSeparator) {
                const regexEscapedSeparator: string = _.replace(
                    separator,
                    /[/\-\\^$*+?.()|[\]{}]/g,
                    '\\$&'
                );
                splits = _.split(text, new RegExp(`(?<=${regexEscapedSeparator})`));
            } else {
                splits = _.split(text, separator);
            }
        } else {
            splits = _.split(text, '');
        }
        return _.filter(splits, s => s !== '');
    }
};

const chapterSplitter = new RecursiveCharacterTextSplitterSeparatorMod({
    separators: ['\n#', '\n\n', '.', '!', '?'], // Chapters, paragraphs, sentences
    chunkSize: 8192,
    keepSeparator: true,
    chunkOverlap: 0,
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
const chapterChunks = await chapterSplitter.splitDocuments(novelText);
const vectorStore = await FaissStoreWithMMR.load(
    storeDirectory,
    embeddings
);
const extractRetriever = vectorStore.asRetriever({
    k: 2,
});
// END OF NOVEL DATA

// SCHEMAS

const EntitySchema = z.object({
    name: z.string().describe('The name of the entity - the primary name by which this entity is known'),
    aliases: z.array(z.string()).optional().describe('Known aliases for the entity, if any, as an array not a string'),
    type: z.string().describe("The type of the entity, including 'Person', 'Location', 'Organization', 'Work of art', 'Theme', 'Concept', 'Vehicle', 'Animal', 'Object'"),
    // .or(z.enum(['Person', 'Location', 'Organization', 'Work of art', 'Theme', 'Concept', 'Vehicle', 'Animal', 'Object']))
    description: z.string().optional().describe('Optional description or distinguishing details'),
});
type Entity = z.infer<typeof EntitySchema>;

const RelationshipSchema = z.object({
    source: z.string().describe('The name of the source entity'),
    target: z.string().describe('The name of the target entity'),
    type: z.string().describe('The type of the relationship'),
    description: z.string().optional().describe('Optional description'),
});
type Relationship = z.infer<typeof RelationshipSchema>;

const EntitiesSchema = z.object({ entities: z.array(EntitySchema).describe('List of entities as an array, not a string') });
const RelationshipsSchema = z.object({ relationships: z.array(RelationshipSchema).describe('List of relationships as an array, not a string') });

// END OF SCHEMAS

// CLASSES

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
        Aliases: ${_.join(entity.aliases, ', ')}
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
            MATCH (entity)
            WHERE entity.embedding IS NOT NULL
            RETURN entity, gds.similarity.cosine(entity.embedding, $entityEmbedding) AS score
            ORDER BY score DESC
            LIMIT toInteger($limit)
        `, { entityEmbedding, limit });

        // Process and return the results
        return _.map(result.records, (record) => {
            const similarEntity = record.get('entity').properties;
            const result = new SimilarEntityResult(
                {
                    name: similarEntity.name,
                    type: similarEntity.type,
                    description: similarEntity.description,
                    aliases: similarEntity.aliases || []
                },
                record.get('score')
            );
            return result;
        });
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
async function updateEntityInNeo4j(existingEntity: Entity, replacementEntity: Entity): Promise<Entity> {
    const session = driver.session();

    try {
        // Calculate the embedding for the replacement entity
        const replacementEntityEmbedding = await generateEntityEmbedding(replacementEntity);

        // Cypher query to update the existing entity with the replacement entity's properties, including the embedding
        // We will add the replacement entity's type as a label and merge the aliases
        await session.run(`
            MATCH (e {name: $existingName, description: $existingDescription})
            WHERE $existingType IN labels(e)
            CALL apoc.create.addLabels(e, [$replacementType]) YIELD node
            SET node.aliases = apoc.coll.toSet(node.aliases + $replacementAliases + [node.name, $replacementName]),
                node.name = $replacementName,
                node.description = $replacementDescription,
                node.embedding = $replacementEmbedding
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

    logger.info(`Updated entity ${existingEntity.name} in Neo4j.\n`);
    return replacementEntity;
}

/**
 * Insert a new entity into the Neo4j database.
 * @param {Entity} entity - The entity to be inserted.
 * @returns {Promise<void>} - A promise that resolves when the insertion is complete.
 */
async function insertEntityIntoNeo4j(entity: Entity): Promise<Entity> {
    const session = driver.session();

    try {
        // Calculate the embedding for the entity
        const entityEmbedding = await generateEntityEmbedding(entity);

        // Cypher query to create a new entity node with the given properties, including the embedding
        await session.run(`
            CREATE (e {
                name: $name,
                description: $description,
                aliases: $aliases,
                embedding: $embedding
            })
            WITH e
            CALL apoc.create.setLabels(e, [$type]) YIELD node
            return node
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

    logger.info(`Inserted entity ${entity.name} into Neo4j.\n`);
    return entity;
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
            logger.info(`Upserting relationship: ${JSON.stringify(relationship)}\n`);
            // Cypher query to merge the relationship if it doesn't already exist
            await session.run(`
                MATCH (a {name: $sourceName}), (b { name: $targetName })
                OPTIONAL MATCH (a)-[r]->(b)
                WHERE type(r) = $type
                WITH a, b, r
                CALL apoc.do.when(
                    r IS NULL,
                    'CALL apoc.create.relationship($a, $relType, $props, $b) YIELD rel RETURN rel',
                    'with $r as r set r += $props RETURN r AS rel',
                    {a: a, b: b, r: r, relType: $type, props: {description: $description}}
                ) YIELD value
                RETURN value.rel
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

    logger.info(`Upserted ${relationships.length} relationships into Neo4j.\n`);
}

// END OF HELPER FUNCTIONS

// CHAINS

const relationshipsLLM = LLM.withStructuredOutput(RelationshipsSchema, { name: 'found_relationships' });
const relationshipExtractionPrompt = ChatPromptTemplate.fromMessages([
    HumanMessagePromptTemplate.fromTemplate(`
You are an expert in entity-relationship extraction. Your task is to identify relationships or connections among the given entities within the provided text.
You will receive a text extract, and a list of entities. Identify any relationships or connections among these entities, specifying the source entity, target entity, relationship type, and an optional description for each relationship.
Relationships can be any type of association between the entities, such as "friends with", is located in", "is a member of", "is related to", etc.
If you do not find any relationships, then call the tool with an empty array. But try hard to find some relationships among the entities - there almost always will be at least one.

When calling a tool, do not JSON stringify the arguments, just pass them as objects or arrays, so for example do this:
\`\`\`
"args": {{"some_array": ["thing1", "thing2"], "other_param": "value"}}
\`\`\`
and not:
\`\`\`
"args": "{{\\"some_array\\": [\\"thing1\\", \\"thing2\\"], \\"other_param\\": \\"value\\"}}" nor
"args": {{ "some_array": "[\\"thing1\\", \\"thing2\\"]", "other_param": "value" }}
\`\`\`

Here is the text extract:
{extract}

Here is the list of entities:
{entities}`),
]);
const relationshipExtractionChain = relationshipExtractionPrompt.pipe(relationshipsLLM);

// Create a new structured output LLM for an array of EntitySchema
const entityRefinementLLM = LLM.withStructuredOutput(EntitiesSchema, { name: 'refined_entities' });
// Define the prompt for refining entities
const entityRefinementPrompt = ChatPromptTemplate.fromMessages([
    HumanMessagePromptTemplate.fromTemplate(`
You are an expert in entity refinement. Your task is to refine a proposed entity using additional context.
You will receive a list of proposed entities, the original extract from which they were extracted, and supplemental extracts for context.
Your goal is to improve the entities' data, particularly their lists of aliases and the names by which they are most commonly known.
Use the supplemental extracts to refine the entities' details and ensure the most accurate and complete representation.

When calling a tool, do not JSON stringify the arguments, just pass them as objects or arrays, so for example do this:
\`\`\`
"args": {{"some_array": ["thing1", "thing2"], "other_param": "value"}}
\`\`\`
and not:
\`\`\`
"args": "{{\\"some_array\\": [\\"thing1\\", \\"thing2\\"], \\"other_param\\": \\"value\\"}}" nor
"args": {{ "some_array": "[\\"thing1\\", \\"thing2\\"]", "other_param": "value" }}
\`\`\`

Proposed Entities:
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
        verboseParsingErrors: true,
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
        verboseParsingErrors: true,
        name: 'insertEntityIntoNeo4j',
        description: 'Insert a new entity into the Neo4j database.',
        schema: z.object({
            entity: EntitySchema
        })
    }
);

const entityAssessmentTools = [updateEntityInNeo4jTool, insertEntityIntoNeo4jTool];
const entityAssessmentToolsNode = new ToolNode(entityAssessmentTools);
const entityAssessmentLLMWithTools = LLM.bindTools(entityAssessmentTools);
const entityAssessmentPrompt = ChatPromptTemplate.fromMessages([
    HumanMessagePromptTemplate.fromTemplate(`
You are an expert in entity assessment and refinement. Your task is to determine if a proposed entities matches any of the similar entities provided.
If a match for a given entity is found, refine the proposed entity by updating its name, aliases, description, and type as necessary, and call the updateEntityInNeo4j tool.
If no match is found for a given entity, call the insertEntityIntoNeo4j tool to add the proposed entity to the database.
Repeat this process for each proposed entity, calling either tool for each proposed entity as needed.

When calling a tool, do not JSON stringify the arguments, just pass them as objects or arrays, so for example do this:
\`\`\`
"args": {{"some_array": ["thing1", "thing2"], "other_param": "value"}}
\`\`\`
and not:
\`\`\`
"args": "{{\\"some_array\\": [\\"thing1\\", \\"thing2\\"], \\"other_param\\": \\"value\\"}}" nor
"args": {{ "some_array": "[\\"thing1\\", \\"thing2\\"]", "other_param": "value" }}
\`\`\`

Here are the proposed entities:
{proposedEntities}

Here are the possibly similar entities:
{similarEntities}`
    )
]);
const entityAssessmentChain = entityAssessmentPrompt.pipe(entityAssessmentLLMWithTools);

// END OF CHAINS

// WORKFLOW STAGE FUNCTIONS

async function extractEntities(state: { novelChunk: { pageContent: string } }): Promise<{ entities: Entity[] }> {
    const pythonScript = 'ner.py'; // Path to your python script

    try {
        const { stdout } = await execa('venv/bin/python', [pythonScript], { input: state.novelChunk.pageContent });
        const entities = JSON.parse(stdout);
        logger.info(chalk.yellow(`Entities extracted: ${JSON.stringify(entities)}\n`));
        return { entities };
    } catch(error) {
        throw new Error(`Failed to extract entities: ${error}`);
    }
}

/**
 * Refine entities using additional context from extracts.
 * @param {Object} state - The current state containing novel chunk and entities.
 * @returns {Promise<Object>} - The updated state with refined entities.
 */
async function refineEntitiesWithContext(state: { novelChunk: { pageContent: string }, entities: Entity[] }): Promise<{ entities: Entity[] }> {
    logger.info(chalk.blue('Refining entities with additional context...\n'));
    const progressBar = bars.create(state.entities.length, 0, { name: state.entities[0]?.name || '' });

    let extracts: ExtractWithContext[] = [];
    for(const entity of state.entities) {
        progressBar.increment({ name: `${entity.name} - extracts` });
        // Retrieve additional extracts for the entity
        extracts = [...extracts, ...(await getExtractsForEntity(entity))];
    }
    progressBar.update({ name: 'Refining entities' });
    // Use entityRefinementChain to refine the entity with additional context
    const refinementResult = await entityRefinementChain.invoke({
        proposedEntity: JSON.stringify(state.entities),
        originalExtract: state.novelChunk.pageContent,
        additionalExtracts: JSON.stringify(extracts),
    });

    progressBar.stop();
    logger.info(chalk.yellow(`Entities refined with context: ${JSON.stringify(_.map(refinementResult.entities, 'name'))}\n`));

    return { entities: refinementResult.entities };
}

/**
 * Further refine entities using similar entities from Neo4j and entityAssessmentChain.
 * @param {Object} state - The current state containing refined entities.
 * @returns {Promise<Object>} - The updated state with further refined entities.
 */
async function furtherRefineEntities(state: { entities: Entity[] }): Promise<{ entities: Entity[] }> {
    logger.info(chalk.blue('Further refining entities using Neo4j...\n'));
    const progressBar = bars.create(state.entities.length, 0, { name: state.entities[0]?.name || '' });

    let similarEntities: SimilarEntityResult[] = [];
    for(const entity of state.entities) {
        progressBar.increment({ name: `${entity.name} - find similar` });
        // Find similar entities in Neo4j
        similarEntities = [...similarEntities, ...(await findSimilarEntitiesInNeo4j(entity))];
    }
    logger.debug(chalk.yellow(`Similar entities found: ${JSON.stringify(similarEntities)}\n`));

    // Use entityAssessmentChain to refine the entity with similar entities
    const assessmentResult = await entityAssessmentChain.invoke({
        proposedEntities: JSON.stringify(state.entities),
        similarEntities: JSON.stringify(_.map(similarEntities, similar => ({
            name: similar.entity.name,
            type: similar.entity.type,
            description: similar.entity.description,
            aliases: similar.entity.aliases
        })))
    });
    const refinedEntities: Entity[] = [];
    try {
        logger.debug(chalk.yellow(`assessmentResult: ${JSON.stringify(assessmentResult)}\n`));
        const toolResult = await entityAssessmentToolsNode.invoke({ messages: [assessmentResult] });
        try {
            logger.info(chalk.yellow(`toolResult: ${toolResult?.messages?.[0]?.content}\n`));
            const refinedEntity = JSON.parse(toolResult?.messages?.[0]?.content || null);
            logger.info(chalk.green(`entityAssessmentToolsNode invocation complete: ${JSON.stringify(refinedEntity || toolResult)}\n`));

            if(refinedEntity) {
                refinedEntities.push(refinedEntity);
            }
        } catch(error) {
            logger.info(chalk.red(`Failed to parse tool result: ${error} for ${JSON.stringify(assessmentResult)}\n`));
        }
    } catch(error) {
        logger.error(chalk.red(`Failed to invoke entityAssessmentToolsNode: ${error} for ${JSON.stringify(assessmentResult)}\n`));
    }

    progressBar.stop();
    logger.info(chalk.yellow(`Entities refined with Neo4j: ${JSON.stringify(_.map(refinedEntities, 'name'))}\n`));

    return { entities: refinedEntities };
}
/**
 * Extract relationships using the relationshipExtractionChain.
 * @param {Object} state - The current state containing further refined entities and novel chunk.
 * @returns {Promise<Object>} - The updated state with extracted relationships.
 */
async function extractRelationships(state: { entities: Entity[], novelChunk: { pageContent: string } }): Promise<{ relationships: Relationship[] }> {
    if(state.entities.length === 0) {
        logger.info(chalk.yellow('No entities to extract relationships from.\n'));
        return { relationships: [] };
    }

    logger.info(chalk.blue('Extracting relationships from the novel chunk...\n'));
    let relationshipsResult = await relationshipExtractionChain.invoke({
        extract: state.novelChunk.pageContent,
        entities: JSON.stringify(state.entities)
    });
    if(_.isString(relationshipsResult)) {
        relationshipsResult = JSON.parse(relationshipsResult);
    }

    logger.info(chalk.green(`Extracted ${relationshipsResult.relationships.length} relationships.\n`));
    return { relationships: relationshipsResult.relationships };
}
/**
 * Save relationships to Neo4j using upsertRelationshipsIntoNeo4j.
 * @param {Object} state - The current state containing relationships.
 * @returns {Promise<void>} - A promise that resolves when the relationships are saved.
 */
async function saveRelationshipsToNeo4j(state: { relationships: Relationship[] }): Promise<void> {
    logger.info(chalk.blue('Saving relationships to Neo4j...\n'));
    await upsertRelationshipsIntoNeo4j(state.relationships);
    logger.info(chalk.green('Relationships saved to Neo4j.\n'));
}
// END OF WORKFLOW STAGE FUNCTIONS

// AGENT WORKFLOW

const ERExtractionAnnotation = Annotation.Root({
    novelMetadata: Annotation<NovelMetadata>,
    novelChunk: Annotation<string>,
    entities: Annotation<Entity[]>,
    relationships: Annotation<Relationship[]>
});

const workflow = new StateGraph(ERExtractionAnnotation)
    .addNode('Setup Metadata', setupMetadata)
    .addNode('Extract Entities', extractEntities)
    .addNode('Refine Entities with Context', refineEntitiesWithContext) // Add this line
    .addEdge(START, 'Setup Metadata')
    .addEdge('Setup Metadata', 'Extract Entities')
    .addEdge('Extract Entities', 'Refine Entities with Context') // Add this line
    .addNode('Further Refine Entities', furtherRefineEntities) // Add this line
    .addEdge('Refine Entities with Context', 'Further Refine Entities') // Add this line
    .addNode('Extract Relationships', extractRelationships)
    .addEdge('Further Refine Entities', 'Extract Relationships')
    .addNode('Save Relationships to Neo4j', saveRelationshipsToNeo4j)
    .addEdge('Extract Relationships', 'Save Relationships to Neo4j')
    .addEdge('Save Relationships to Neo4j', END);

const app = workflow.compile();

// END OF AGENT WORKFLOW

// // Now run:
const totalProgress = bars.create(chapterChunks.length, 0);
for(const novelChunk of chapterChunks) {
    totalProgress.update({ name: `${JSON.stringify(novelChunk.pageContent).substring(0, 48)}...(${novelChunk.pageContent.length})` });
    const initialState = {
        novelChunk: novelChunk,
    };

    for await (const output of await app.stream(initialState, { streamMode: 'values', recursionLimit: 50 })) {
        logger.info(chalk.green(`Entities: ${output?.entities?.length || 0} Relationships: ${output?.relationships?.length || 0}\n`));
    }
    totalProgress.increment();
}

bars.stop();
await driver.close();
