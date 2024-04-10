import chalk from 'chalk';

import { OllamaEmbeddings } from '@langchain/community/embeddings/ollama';
import { Ollama } from '@langchain/community/llms/ollama';
import { PromptTemplate } from '@langchain/core/prompts';
import neo4j from 'neo4j-driver';
import { Neo4jGraph } from '@langchain/community/graphs/neo4j_graph';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { TokenTextSplitter } from 'langchain/text_splitter';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { Node, Relationship, GraphDocument, } from './node_modules/@langchain/community/dist/graphs/graph_document.js';
import { JsonOutputFunctionsParser } from 'langchain/output_parsers';

import _ from 'lodash';
import cliProgress from 'cli-progress';
import { parse as jsonParse } from 'tolerant-json-parser';

const url = process.env.NEO4J_URI;
const username = process.env.NEO4J_USER;
const password = process.env.NEO4J_PASSWORD;

const book = 'Christmas Town draft 2';
const graph = await Neo4jGraph.initialize({ url, username, password });

const loader = new TextLoader(`novels/${book}.txt`);
const docs = await loader.load();

const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 8192,
    chunkOverlap: 128,
});

const splits = await splitter.splitDocuments(docs);
console.log(chalk.blueBright(`Split into ${splits.length} chunks.`));

const [B_INST, E_INST] = ['<s>[INST] ', ' [/INST] \`\`\`json'];
const FIND_ENTITIES_PROMPT = `You are an expert at helping to build knowledge graphs as a representation of information contained in text extracts from a novel.

Your job is to identify all the most important entities from a particular extract of the novel, and to output a list of those entities. Be as comprehensive as possible, and include ALL the major entities. Avoid pronouns as entities - try and resolve who or what the pronoun refers to, and use that as the entity instead.

You must only include entities which are present in the text - do not make anything up. If there are no relevant entities, then just return nothing. Do not include minor entities which do not appear to be important.

For each entity, you need to classify that entity into a type or category. Some suggested types will be provided, and you should use those where relevant, but you can create your own type if none of the suggestions is appropriate for an important entity that you find.

Another expert will later be analyzing the relationships among these entities, so please capture all the important entities so we don't miss any important relationships later. Do not include the relationship info itself in what you return though - that'll be done later.

The overall knowledge graph will be used to analyze the novel and identify important characters, plot elements, story progression and many other aspects of the novel, so please make it as complete as possible.

If an entity is known by more than one name, include an entity for each name.

You should return your results in JSON format like this:
\`\`\`
{{ "nodes": [
        {{ id: ENTITY, type: TYPE }},
        {{ ... }}
    ]
}}
\`\`\`

Example:
---
Suggested types: ["Person", "Job"]
Novel extract: Alice is a lawyer and is 25 years old and Robert is her roommate since 2001. Bob works as a journalist. Alice owns a the webpage www.alice.com and Bob owns the webpage www.bob.com. Bob and Alice are siblings.
\`\`\`
{{ "nodes": [
        {{ "id": "Alice", "type": "Person" }},
        {{ "id": "lawyer", "type": "Job" }},
        {{ "id": "Robert", "type": "Person" }},
        {{ "id": "journalist", "type": "Job" }},
        {{ "id": "alice.com", "type": "Webpage" }},
        {{ "id": "bob.com", "type": "Webpage" }}
    ]
}}
\`\`\`
---

If there are no relevant nodes, return this, by itself with no pre-amble:
\`\`\`
{{ "nodes": [] }}
\`\`\`

BEGIN!

Suggested Types: {labels}
Novel extract: {data}`;
const findEntitiesPrompt = PromptTemplate.fromTemplate(`${B_INST}${FIND_ENTITIES_PROMPT}${E_INST}`);

const FIND_RELATIONSHIPS_PROMPT = `You are an expert at helping to build knowledge graphs as a representation of information contained in text extracts from a novel.

Your job is to identify all the the important relationships among a provided set of entities within a particular section of the novel, and to output a list of relationships.

You must only include relationships and entities which are present in the text - do not make anything up. If there are no relevant relationships, then just return nothing.

For each provided entity, you must include at least one relationship between that entity and another entity.

For each Relationship, you need to identify a label or type for that relationship. Some suggested types are provided; use those where appropriate but you can create new types if the relationship isn't adequately described by any of the suggestions.

Another expert will be analyzing the knowledge graph later, so it's very important to capture all the important relationships so we don't miss any important aspects of the novel. Focus especially on capturing relationships among the people.

The overall knowledge graph will be used to analyze the novel and identify important characters, plot elements, story progression and many other aspects of the novel, so please make it as complete as possible.

Try hard to find at least one relationship, and ideally more than one, for each of the entities provided, but do not make anything up, the relationship must exist within the provided novel extract. Every entity should have at least one relationship connecting it to the graph. Check each pair of entities to see if they have a relationship.

If an entity is known by more than one name, include an "Alias or nickname" relationship between each name the entity is known by.

You should return your results in JSON format like this:
\`\`\`
{{ "relationships": [
        {{ "source": "Entity1_id", "type": "Relationship Type", "target": "Entity2_id" }},
        {{ ... }}
    ]
}}
\`\`\`

Property values can only be of primitive types or arrays thereof.

Example:
---
Entities: {{ "nodes": [
        {{ "id": "Alice", "type": "Person" }},
        {{ "id": "lawyer", "type": "Job" }},
        {{ "id": "Robert", "type": "Person" }},
        {{ "id": "journalist", "type": "Job" }},
        {{ "id": "alice.com", "type": "Webpage" }},
        {{ "id": "bob.com", "type": "Webpage" }}
    ]
}}
Novel extract: Alice is a lawyer and is 25 years old and Robert is her roommate since 2001. Bob works as a journalist. Alice owns a the webpage www.alice.com and Bob owns the webpage www.bob.com. Bob and Alice are siblings.
\`\`\`
{{ "relationships": [
        {{ "source": "Alice", "type": "roommate", "target": "Robert" }},
        {{ "source": "Alice", "type": "works as", "target": "lawyer" }},
        {{ "source": "Robert", "type": "roommate", "target": "Alice" }},
        {{ "source": "Robert", "type": "works as", "target": "journalist" }},
        {{ "source": "Alice", "type": "owns", "target": "alice.com" }},
        {{ "source": "Robert", "type": "owns", "target": "bob.com" }},
        {{ "source": "Robert", "type": "has sister", "target": "Alice" }},
        {{ "source": "Alice", "type": "has brother", "target": "Robert" }},
    ]
}}
\`\`\`
---

BEGIN!

Suggested relationships: {labels}
Entities: {entities}
Novel extract: {data}`;
const findRelationshipsPrompt = PromptTemplate.fromTemplate(`${B_INST}${FIND_RELATIONSHIPS_PROMPT}${E_INST}`);

const embeddings = new OllamaEmbeddings({ model: 'nomic-embed-text', numCtx: 2048, baseUrl: 'http://127.0.0.1:11435' });

// Mistral 7b-instruct in the cloud on amazon - cheap and fast, but not so smrt; 32k context 8k outputs
// const llm = new Bedrock({
//     model: "mistral.mistral-7b-instruct-v0:2", // You can also do e.g. "anthropic.claude-v2"
//     region: "us-west-2",
//     temperature: 0,
//     maxTokens: 8192,
// });

// mistral-instruct-v0.2-2x7b-moe has 32k context, instruct model
// const llm = new Ollama({ model: 'cas/mistral-instruct-v0.2-2x7b-moe', temperature: 0, numCtx: 32768, format: 'json', raw: true });

// mistral-instruct-v0.2-2x7b-moe has 32k ctx training, 2k context runtime suggested, instruct model
const llm = new Ollama({ model: 'mistral:instruct', temperature: 0, numCtx: 32768, format: 'json', raw: true });

// mix-crit is from mixtral:8x7b-instruct-v0.1-q5_K_M - 32k context, instruct model
// const llm = new Ollama({ model: 'mix-crit', temperature: 0, numCtx: 32768, format: 'json', raw: true });

// Same guy, but in the cloud - faster but more $
// const llm = new Bedrock({
//     model: "mistral.mixtral-8x7b-instruct-v0:1", // You can also do e.g. "anthropic.claude-v2"
//     region: "us-west-2",
//     temperature: 0,
//     maxTokens: 4096,
// });

const findEntitiesChain = findEntitiesPrompt.pipe(llm);
const findRelationshipsChain = findRelationshipsPrompt.pipe(llm);

const bar = new cliProgress.SingleBar({ barsize: 80, format: '{bar} {value}/{total} splits | {percentage}% | Time: {duration_formatted} | ETA: {eta_formatted}'}, cliProgress.Presets.shades_classic);
bar.start(splits.length*2, 0);
for(let split of splits) {
    // console.log(chalk.grey(split.pageContent));

    let entities = await findEntitiesChain.invoke({
        data: split.pageContent,
        labels: `["Person", "Occupation", "Location", "Vehicle"]`,
    });
    // console.log(chalk.greenBright(entities));
    try {
        entities = jsonParse(entities);
    } catch {
        continue;
    } finally {
        bar.increment();
    }

    if (entities.length == 0) {
        bar.increment(); // Increment for skipped relation check
        continue;
    }

    let relationships = await findRelationshipsChain.invoke({
        data: split.pageContent,
        entities: JSON.stringify(entities),
        labels: `["Owns", "Alias or nickname", "Works as", "Mother of", "Father of", "Daughter of", "Son of", "Lives in", "Located in", "Feels", "Hates", "Likes", "In love with", "Friends with", "Takes care of"]`
    });

    // console.log(chalk.greenBright(relationships));
    try {
        relationships = jsonParse(relationships);
    } catch {
        continue;
    } finally {
        bar.increment();
    }

    entities = _.chain(entities.nodes)
        .map(e => new Node({
            id: e.id,
            type: e.type && e.type.replaceAll(' ', '_') || undefined,
            properties: e.properties
        }))
        .filter('id')
        .filter('type')
        .value();
    if (entities.length == 0) { continue; }

    relationships = _.chain(relationships.relationships)
        .map(r => new Relationship({
            source: _.find(entities, { id: r.source }),
            type: r.type && r.type.replaceAll(' ', '_') || undefined,
            target: _.find(entities, { id: r.target }),
            properties: r.properties
        }))
        .filter('source')
        .filter('type')
        .filter('target')
        .value();

    let graphDoc = new GraphDocument({ nodes: entities, relationships, source: split });
    await graph.addGraphDocuments([graphDoc]);
}
bar.stop();
await graph.close();
