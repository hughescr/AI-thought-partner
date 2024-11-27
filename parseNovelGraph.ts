import { qwen25_32bLLM as structureableLLM } from './lib/LLMs.ts';
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

const loader: TextLoader = new TextLoader(`novels/${book}.md`);
const novelText = await loader.load();
const textChunks = textSplitter.splitText(novelText);

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

async function extractEntitiesAndRelationships(text: string) {
    const response = await structuredLlm.invoke({
        input: text
    });
    return response;
}

async function upsertEntities(session: Session, entities: Array<{ name: string; type: string; description: string }>) {
    for (const entity of entities) {
        await session.run(
            'MERGE (e:Entity {name: $name, type: $type, description: $description})',
            entity
        );
    }
}

async function upsertRelationships(session: Session, relationships: Array<{ source: string; target: string; type: string; description?: string }>) {
    for (const relationship of relationships) {
        await session.run(
            'MATCH (a:Entity {name: $source}), (b:Entity {name: $target}) MERGE (a)-[:' + relationship.type + ' {description: $description}]->(b)',
            relationship
        );
    }
}

const session = driver.session();
try {
    for (const chunk of textChunks) {
        const { entities, relationships } = await extractEntitiesAndRelationships(chunk);
        await upsertEntities(session, entities);
        await upsertRelationships(session, relationships);
    }
} finally {
    await session.close();
    await driver.close();
}
