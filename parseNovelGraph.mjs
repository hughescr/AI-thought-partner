import { OllamaEmbeddings } from '@langchain/ollama';
import { CacheBackedEmbeddings } from "langchain/embeddings/cache_backed";
import { InMemoryStore } from "langchain/storage/in_memory";
import { ChatOllama } from '@langchain/ollama';
import { ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate } from '@langchain/core/prompts';
import { TextLoader } from 'langchain/document_loaders/fs/text';
// import { SemanticTextSplitter } from './lib/SemanticChunker.mjs';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import neo4j from 'neo4j-driver';

import z from 'zod';
import _ from 'lodash';
import chalk from 'chalk';
import cliProgress from 'cli-progress';

const neo4jURL = process.env.NEO4J_URI;
const neo4jUsername = process.env.NEO4J_USER;
const neo4jPassword = process.env.NEO4J_PASSWORD;

const book = 'Christmas Town beta';
const loader = new TextLoader(`novels/${book}.md`);
const novelText = await loader.load();

const coreEmbeddings = new OllamaEmbeddings({ model: 'nomic-embed-text', numCtx: 2048 });
// const coreEmbeddings = new OllamaEmbeddings({ model: 'mxbai-embed-large', numCtx: 512 });
// const coreEmbeddings = new OllamaEmbeddings({ model: 'mistral-nemo:12b-instruct-2407-q8_0', numCtx: 32768 });
// const coreEmbeddings = new OllamaEmbeddings({ model: 'llama3.1:8b-instruct-q8_0', numCtx: 2048 });

const store = new InMemoryStore();
const embeddings = CacheBackedEmbeddings.fromBytesStore(
    coreEmbeddings,
    store,
    {
        namespace: coreEmbeddings.modelName,
    }
)

const commonOptions = { temperature: 0, seed: 19740822, keepAlive: '15m' };
const commonOptionsJSON = { ...commonOptions, temperature: 0, format: 'json' };
const commonOptions8k = { numCtx: 8 * 1024, ...commonOptions };
const commonOptions8kJSON = { ...commonOptions8k, ...commonOptionsJSON };
const commonOptions32k = { numCtx: 32 * 1024, ...commonOptions };
const commonOptions32kJSON = { ...commonOptions32k, ...commonOptionsJSON };
const commonOptions64k = { numCtx: 64 * 1024, ...commonOptions };
const commonOptions64kJSON = { ...commonOptions64k, ...commonOptionsJSON };
const commonOptions128k = { numCtx: 128 * 1024, ...commonOptions };
const commonOptions128kJSON = { ...commonOptions128k, ...commonOptionsJSON };
const commonOptions256k = { numCtx: 256 * 1024, ...commonOptions };
const commonOptions256kJSON = { ...commonOptions256k, ...commonOptionsJSON };

// mistral-nemo has 1024k ctx
// Prompt parse: ~500 t/s; generation: ~40 t/s
const nemo_12bLLMChat = new ChatOllama({ model: 'mistral-nemo:12b-instruct-2407-q8_0', ...commonOptions64k });
const nemo_12bLLMJSON = new ChatOllama({ model: 'mistral-nemo:12b-instruct-2407-q8_0', ...commonOptions32kJSON });

// mistral-large has 1024k ctx
// Prompt parse: ~slow t/s; generation: ~slow t/s
const mistralLargeLLMChat = new ChatOllama({ model: 'mistral-large:latest', ...commonOptions64k });
const mistralLargeLLMJSON = new ChatOllama({ model: 'mistral-large:latest', ...commonOptions32kJSON });

// llama3.1 has 128k ctx
// Prompt parse: ~500 t/s; generation: ~40 t/s
const llama3_8bLLMChat = new ChatOllama({ model: 'llama3.1:8b-instruct-q8_0', ...commonOptions64k });
const llama3_8bLLMJSON = new ChatOllama({ model: 'llama3.1:8b-instruct-q8_0', ...commonOptions32kJSON });

// process_novel.js

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

// Create a structured LLM with the schema
const structuredLlm = mistralLargeLLMChat.withStructuredOutput(OutputSchema);

// Initialize the SemanticTextSplitter with adjusted chunk size
const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 8 * 1024, // Adjust as needed for about a chapter's worth
    keepSeparator: true,
    separators: ['##'],
});

// Define the system prompt for entity extraction
const systemPrompt = SystemMessagePromptTemplate.fromTemplate(`You are an AI assistant that extracts entities and relationships from text extracts of a novel.
For the given text, extract all entities (people, locations, organizations) and relationships.

Provide sufficient description to dis-ambiguate each entity.

You must *always* include a name, and type for each entity. You should also include a description; descriptions will be super helpful - make your very best effort to describe each entity in a sentence or two.

You must *always* include a source, target, and type for each relationship. Try to include at least one relationship for each new entity found.

You must *always* respond with a tool call; if you found no entities or relationships, make a tool call and pass in empty arrays.

Your output will be used to analyze the text later. It's extremely important that you provide accurate and consistent information.
It's also important that you list all entities and relationships in the text, do not leave anything out.
Do not make anything up. All of the entities and relationships should be based on the text provided.

Remember to always make a tool call, so your response should always be JSON like this, without any markdown quoting:

\`\`\`
"tool_calls":[{{"function":{{"name":"extract","arguments":{{"entities":[...],"relationships":[...]}}}}}}]
\`\`\`
`);
// Combine the system prompt and the chunk
const prompt = ChatPromptTemplate.fromMessages([
    systemPrompt,
    HumanMessagePromptTemplate.fromTemplate(`{chunk}`),
]);

// Initialize Neo4j driver
const driver = neo4j.driver(
    neo4jURL,
    neo4j.auth.basic(neo4jUsername, neo4jPassword) // Replace with your Neo4j password
);

// Main async function
(async () => {
    try {
        // Split the novel text into chunks
        const chunks = await textSplitter.splitDocuments(novelText);

        // Initialize the progress bar
        const progressBar = new cliProgress.SingleBar({
            format: 'Processing chunks |' + chalk.cyan('{bar}') + '| {percentage}% || Chunk: {value}/{total} || Elapsed: {duration_formatted} Rate: {speed}s/it ETA: {eta_formatted}',
            barCompleteChar: '\u2588',
            barIncompleteChar: '\u2591',
            fps: 1,
            hideCursor: true,
        });
        progressBar.start(chunks.length, 0, { speed: 'N/A' });

        for (const [index, chunk] of chunks.entries()) {
            // Update progress bar
            progressBar.update(index + 1, { speed: Math.round(10*Math.round((Date.now() - progressBar.startTime) / 1000)/progressBar.value)/10 });

            // Extract entities and relationships using the structured LLM
            let response;
            try {
                response = await prompt.pipe(structuredLlm).invoke({chunk: chunk.pageContent});
            } catch (error) {
                console.error(chalk.red(`\nFailed to process chunk ${index + 1}:`), error);
                continue;
            }

            // The response is already parsed according to the schema
            const data = response;

            const entities = data.entities || [];
            const relationships = data.relationships || [];

            // Upsert entities into Neo4j with advanced entity resolution
            let entitiesWithIDs = {};
            if(entities.length > 0) {
                const result = await upsertEntitiesWithResolution(driver, entities, embeddings);
                entitiesWithIDs = _.zipObject(_.map(entities, entity => entity.name),
                                        _.map(entities, (entity, index) => ({...entity, id: result[index]}))
                                    );
            }

            // Create relationships in Neo4j
            if(relationships.length > 0) {
                await createRelationshipsWithResolution(driver, relationships, embeddings, entitiesWithIDs);
            }
        }
        // Stop the progress bar after processing
        progressBar.stop();

        console.log(chalk.green('Processing complete!'));
    } catch (error) {
        console.error(chalk.red('An error occurred:'), error);
    } finally {
        // Close the Neo4j driver
        await driver.close();
    }
})();

// Function to upsert multiple entities with batching and Neo4j GDS for similarity search
async function upsertEntitiesWithResolution(driver, entities, embeddings) {
    try {
        // Prepare texts for embeddings
        const entityTexts = entities.map(
            (entity) => `${entity.name}. Type: ${entity.type}. Description: ${entity.description || entity.name}`
        );

        // Generate embeddings in batches
        const entityEmbeddings = await embeddings.embedDocuments(entityTexts);

        // For each entity, perform similarity search in Neo4j
        const upsertedEntityIDs = await Promise.all(_.map(entities, async (entity, index) => {
            const session = driver.session();
            try {
                const entityEmbedding = entityEmbeddings[index];

                // Store the embedding as an array property
                const embeddingArray = Array.from(entityEmbedding);

                // Perform similarity search using GDS
                const similarEntities = await findSimilarEntities(driver, embeddingArray, entity.name);

                if (similarEntities.length > 0) {
                    const matchedEntity = similarEntities[0].node.properties;
                    // Merge with existing entity
                    console.log(chalk.yellow(`Found similar entity: ${entity.name} -> ${matchedEntity.name} (${similarEntities[0].similarity})`));
                    await session.run(
                        `
            MATCH (e {id: $existingId})
            SET e.description = CASE WHEN $newDescription = '' THEN e.description ELSE $newDescription END
            RETURN e
            `,
                        {
                            existingId: matchedEntity.id,
                            newDescription: matchedEntity.description + entity.description,
                        }
                    );
                    return matchedEntity.id;
                } else {
                    // Create new entity with embedding
                    const response = await session.run(
                        `
            CREATE (e:${entity.type.toUpperCase().replace(/(-|\s)+/g, '_')} {id: randomUUID(), name: $name, description: $description, embedding: $embedding})
            RETURN e
            `,
                        {
                            name: entity.name,
                            type: entity.type,
                            description: entity.description || '',
                            embedding: embeddingArray,
                        }
                    );
                    return response.records[0].get('e').properties.id;
                }
            }
            finally {
                await session.close();
            }
        }));
        return upsertedEntityIDs;
    } catch (error) {
        console.error('Failed to upsert entities:', error);
    } finally {
    }
}

// Function to find similar entities using Neo4j GDS
async function findSimilarEntities(driver, embedding, name, similarity = 0.9) {
    const session = driver.session();
    try {
        // Create an in-memory graph for similarity search
        // Note: For large datasets, consider creating a persistent graph catalog

        // Define parameters
        const params = {
            embedding,
            similarity,
            name,
        };

        // Run the similarity search using cosine similarity
        const result = await session.run(
            `
      WITH $embedding AS queryEmbedding
      MATCH (e)
      WHERE e.embedding IS NOT NULL
      WITH e, gds.similarity.cosine(queryEmbedding, e.embedding) AS similarity
      WHERE similarity > $similarity OR e.name = $name
      RETURN e AS node, similarity
      ORDER BY similarity DESC
      LIMIT 1
      `,
            params
        );

        return result.records.map((record) => ({
            node: record.get('node'),
            similarity: record.get('similarity'),
        }));
    } catch (error) {
        console.warn(chalk.yellow('Failed to find similar entities:'), error);
        return [];
    } finally {
        await session.close();
    }
}

// Function to create multiple relationships with entity resolution
async function createRelationshipsWithResolution(driver, relationships, embeddings, entitiesWithIDs) {
    const session = driver.session();
    try {
        // Extract unique entity names from relationships
        const entityNames = [
            ...new Set(relationships.flatMap((rel) => [rel.source, rel.target])),
        ];

        // Resolve entities
        const resolvedEntities = {};
        const resolveBar = new cliProgress.SingleBar({
            format: 'Resolving entities |' + chalk.magenta('{bar}') + '| {percentage}% || Entity: {value}/{total} || ETA: {eta_formatted}',
            barCompleteChar: '\u2588',
            barIncompleteChar: '\u2591',
            hideCursor: true,
        });
        resolveBar.start(entityNames.length, 0);

        for (const [i, entityName] of entityNames.entries()) {
            let resolvedEntity = entitiesWithIDs[entityName];
            if(!resolvedEntity) {
                // Resolve entity using embeddings
                resolvedEntity = await resolveEntity(driver, entityName, embeddings);
            }
            if (resolvedEntity) {
                resolvedEntities[entityName] = resolvedEntity;
            }
            resolveBar.update(i + 1);
        }
        resolveBar.stop();

        const relationshipBar = new cliProgress.SingleBar({
            format: 'Creating relationships |' + chalk.yellow('{bar}') + '| {percentage}% || Relationship: {value}/{total} || ETA: {eta_formatted}',
            barCompleteChar: '\u2588',
            barIncompleteChar: '\u2591',
            hideCursor: true,
        });
        relationshipBar.start(relationships.length, 0);

        for (const [i, relationship] of relationships.entries()) {
            const sourceEntity = resolvedEntities[relationship.source];
            const targetEntity = resolvedEntities[relationship.target];

            if (sourceEntity && targetEntity) {
                await session.run(
                    `
          MATCH (source {id: $sourceId})
          MATCH (target {id: $targetId})
          MERGE (source)-[r:${relationship.type.toUpperCase().replace(/(-|\s)+/g, '_')}]->(target)
          ON CREATE SET r.description = $description
          RETURN r
          `,
                    {
                        sourceId: sourceEntity.id,
                        targetId: targetEntity.id,
                        description: relationship.description || '',
                    }
                );
            } else {
                console.warn(chalk.yellow(`Could not resolve entities for relationship: ${relationship.source} -> ${relationship.target}`));
            }

            relationshipBar.update(i + 1);
        }
        relationshipBar.stop();
    } catch (error) {
        console.error(chalk.red('Failed to create relationships:'), error);
    } finally {
        await session.close();
    }
}

// Function to resolve an entity by name using embeddings and Neo4j GDS
async function resolveEntity(driver, entityName, embeddings) {
    const session = driver.session();
    try {
        // Prepare the text for embedding
        const entityText = `${entityName}. Type: Unknown. Description: ${entityName}`;

        // Generate embedding for the entity name
        const [entityEmbedding] = await embeddings.embedDocuments([entityText]);
        const embeddingArray = Array.from(entityEmbedding);

        // Perform similarity search using GDS
        const similarEntities = await findSimilarEntities(driver, embeddingArray, entityName, 0.5);

        if (similarEntities.length > 0) {
            return similarEntities[0].node.properties;
        } else {
            // Entity not found, create a new one
            const response = await upsertEntityWithResolution(driver, { name: entityName, type: 'Unknown' }, embeddings);
            return { id: response, name: entityName, type: 'Unknown' };
        }
    } catch (error) {
        console.error(chalk.red('Failed to resolve entity:'), error);
        return null;
    } finally {
        await session.close();
    }
}

// Function to upsert a single entity with resolution
async function upsertEntityWithResolution(driver, entity, embeddings) {
    // Reuse the upsertEntitiesWithResolution function with a single entity
    return (await upsertEntitiesWithResolution(driver, [entity], embeddings))[0];
}
