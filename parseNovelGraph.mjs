import chalk from 'chalk';

import { OllamaEmbeddings } from '@langchain/community/embeddings/ollama';
import { CacheBackedEmbeddings } from "langchain/embeddings/cache_backed";
import { InMemoryStore } from "langchain/storage/in_memory";
import { ChatOllama } from '@langchain/community/chat_models/ollama';
import { ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate } from '@langchain/core/prompts';
import { Neo4jGraph } from '@langchain/community/graphs/neo4j_graph';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { TokenTextSplitter } from 'langchain/text_splitter';
import { Node, Relationship, GraphDocument, } from './node_modules/@langchain/community/dist/graphs/graph_document.js';
import { RunnableSequence, RunnablePassthrough } from '@langchain/core/runnables';
import { StringOutputParser, JsonOutputParser } from 'langchain/schema/output_parser';

import _ from 'lodash';
import cliProgress from 'cli-progress';

const url = process.env.NEO4J_URI;
const username = process.env.NEO4J_USER;
const password = process.env.NEO4J_PASSWORD;

const book = 'Christmas Town draft 2';
const graph = await Neo4jGraph.initialize({ url, username, password });

const loader = new TextLoader(`novels/${book}.txt`);
const docs = await loader.load();

const commonOptions = { temperature: 0, seed: 19740822 };
const commonOptionsJSON = { format: 'json' };
const commonOptions8k = { numCtx: 8 * 1024, ...commonOptions };
const commonOptions8kJSON = { ...commonOptions8k, ...commonOptionsJSON };
const commonOptions32k = { numCtx: 32 * 1024, ...commonOptions };
const commonOptions32kJSON = { ...commonOptions32k, ...commonOptionsJSON };
const commonOptions64k = { numCtx: 64 * 1024, ...commonOptions };
const commonOptions64kJSON = { ...commonOptions64k, ...commonOptionsJSON };

// mistral:7b-instruct-v0.2-q8_0 has 32k training ctx but ollama sets it to 2k so need to override that here
// Prompt parse: ~600-725 t/s; generation: ~18 t/s
const mistralLLMChat = new ChatOllama({ model: 'mistral:7b-instruct-v0.2-q8_0', ...commonOptions32k });
const mistralLLMJSON = new ChatOllama({ model: 'mistral:7b-instruct-v0.2-q8_0', ...commonOptions32kJSON });

// llama3 llama3:8b-instruct-Q8_0 has 8k ctx
// Prompt parse: ~700 t/s; generation: ~11 t/s
const llama3LLMChat = new ChatOllama({ model: 'llama3:8b-instruct-Q8_0', ...commonOptions8k });
const llama3LLMJSON = new ChatOllama({ model: 'llama3:8b-instruct-Q8_0', ...commonOptions8kJSON });

// mixtral:8x7b-instruct-v0.1-q8_0 - 32k context
// Prompt parse: ~150-200 t/s; generation: ~20-25 t/s
const mixtral7BLLMChat = new ChatOllama({ model: 'mixtral:8x7b-instruct-v0.1-q8_0', ...commonOptions32k });
const mixtral7BLLMJSON = new ChatOllama({ model: 'mixtral:8x7b-instruct-v0.1-q8_0', ...commonOptions32kJSON });

// mixtral:8x22b-instruct-v0.1-q4_0 - 64k training context but ollama sets it to 2k, has special tokens for tools and shit
// Prompt parse: ~60-80 t/s; generation ~11-12 t/s
const mixtral22bLLMChat = new ChatOllama({ model: 'mixtral:8x22b-instruct-v0.1-q4_0', ...commonOptions64k });
const mixtral22bLLMJSON = new ChatOllama({ model: 'mixtral:8x22b-instruct-v0.1-q4_0', ...commonOptions64kJSON });

// llama3 llama3:70b-instruct-q8_0 has 8k ctx
// Prompt parse: ~60 t/s; generation ~4 t/s
const llama3_70bLLMChat = new ChatOllama({ model: 'llama3:70b-instruct-q8_0', ...commonOptions8k });
const llama3_70bLLMJSON = new ChatOllama({ model: 'llama3:70b-instruct-q8_0', ...commonOptions8kJSON });


const splitter = new TokenTextSplitter({
    chunkSize: 4096, // Tokens!
    chunkOverlap: 128,
});

const splits = await splitter.splitDocuments(docs);
console.log(chalk.blueBright(`Split into ${splits.length} chunks.`));

const findEntitiesPrompt = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(`# System Overview
You are an expert at helping to build knowledge graphs as a representation of information contained in text extracts from a novel.
Your job is to identify all the named entities from a particular extract of the novel, and to output a list of those entities in JSON format.

## Important Considerations
  - Be as comprehensive as possible, and include ALL of the entities from this extract in your result, even if they were already known.
  - Resolve pronouns to entities: try and resolve who or what the pronoun refers to, and use that as the entity instead.
  - You will be given a list of previous entities. If any of these exist in the extract, include them in the output with the same name, updating any information on known aliases if you have found an alias.
  - If there's an entity in the entities which is not in the extract, you can omit it from the output.
  - Entities should not simply be nouns or noun phrases or descriptors in the extract. For example in the sentence "John is a small man", "John" is an entity, but "small man" is not. In the sentence "The principal was a small, balding man.", the principal is an entity, but ideally he would be labeled with his name (if it is known); in any case "small balding man" is not an entity.
  - If an important concept in the story does not have a specific name though, it should be considered an entity. For example, if all the characters go to school together, but the school is never named, then "School" would be an appropriate entity.

## Do not invent things
You must only include entities which are present in the text - do not make anything up.
If there are no entities in the extract, then return an empty array.

## Classification
For each entity, you need to classify that entity into a type or category.
Some suggested types will be provided, and you should use those where relevant, but you can create your own type if none of the suggestions is appropriate for an important entity that you find.
Do not create a new type if one of the suggested types will suffice. For example if there's a "Person" type, then all people should be classified as "Person", do not make up another type like "Mother".

## Aliases and Disambiguation
If an entity is known by more than one name, try to create only a single entity, but include a property to indicate the list of known aliases.
Be especially thoughtful about aliases for previous entities. If you find a new alias for an existing entity, then keep that entity's name and set the alias property for that entity.

You should return your results in JSON format like this:
\`\`\`
{{"entities":[
    {{ "id": the name of the entity, "type": the classification or type of the entity, "properties": {{ "aliases": [ list of known aliases for this entity ] }} }},
    {{ ... }}
]}}
\`\`\`

If there are no relevant entities, return an empty array, by itself with no pre-amble:
\`\`\`
{{"entities":[]}}
\`\`\``),
//     HumanMessagePromptTemplate.fromTemplate(`Suggested types: ["Person","Job"]
// Previous entities: ["Bob"]
// Novel extract: Alice is a lawyer and is 25 years old and Robert is her roommate since 2001.
// Bob works as a journalist.
// Alice owns a the webpage www.alice.com and Bob owns the webpage www.bob.com.
// Bob and Alice are siblings.`),
//     AIMessagePromptTemplate.fromTemplate(`{{"entities":[{{"id":"Alice","type":"Person"}},{{"id":"lawyer","type":"Job"}},{{"id":"Bob","type":"Person","properties":{{"aliases":["Robert","Bob"]}}}},{{"id":"journalist","type":"Job"}},{{"id":"alice.com","type":"Webpage"}},{{"id":"bob.com","type":"Webpage"}}]}}`),
//     HumanMessagePromptTemplate.fromTemplate(`Suggested types: ["Person"]
// Previous entities: []
// Novel extract: That morning, Robert put on his shoes, and went to the park.
// David saw him there. "Hi, Bob!", said Dave.`),
//     AIMessagePromptTemplate.fromTemplate(`{{"entities":[{{"id":"Robert","type":"Person","properties":{{"aliases":["Robert","Bob"]}}}},{{"id":"shoes","type":"Clothing"}},{{"id":"park","type":"Location"}},{{"id":"David","type":"Person","properties":{{"aliases":["David","Dave"]}}}}]}}`),
    HumanMessagePromptTemplate.fromTemplate(`Suggested Types: {labels}
Previous entities: {entities}
Novel extract: {data}`)
]);

const findRelationshipsPrompt = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(`# System Overview
You are an expert at helping to build knowledge graphs as a representation of information contained in text extracts from a novel.
Your job is to identify all the the important relationships among a provided set of entities within a particular section of the novel, and to output a list of relationships.

## Important Considerations
If any provided entity is referenced in the novel extract, then you should include any relationship between that entity and others in your output.
Try hard to find at least one relationship, and ideally more than one, for each of the entities provided, but do not make anything up, the relationship must exist within the provided novel extract.
Check each pair of entities to see if they have a relationship expressed in the novel extract.
Try and identify every relationship that you find in the extract among any of the listed entities.

## Do not invent things
You must only include relationships and entities which are present in the text - do not make anything up. If there are no relevant relationships, then just return nothing.

## Classification of relationships
For each Relationship, you need to identify a label or type for that relationship.
Some suggested types are provided; use those where appropriate but you can create new types if the relationship isn't adequately described by any of the suggestions.
Do not create a new type if one of the suggested types will suffice. For example, do not create "is the son of" if "son of" already exists; just use the existing "son of".

## Aliases, nicknames, and disambiguation
If an entity is known by more than one name, include an "alias of" relationship between each name the entity is known by.

You should return your results in JSON format like this:
\`\`\`
{{"relationships":[
    {{ "source": "Entity1", "type": "Relationship Type", "target": "Entity2" }},
    {{ ... }}
]}}
\`\`\``),
//     HumanMessagePromptTemplate.fromTemplate(`Suggested relationships: ["owns","works as","has sister","has brother","has child","has parent"]
// Entities: ["Alice","lawyer","Robert","journalist","alice.com","bob.com"]
// Novel extract: Alice is a lawyer and is 25 years old and Robert is her roommate since 2001.
// Bob works as a journalist. Alice owns a the webpage www.alice.com and Bob owns the webpage www.bob.com.
// Bob and Alice are siblings.`),
//     AIMessagePromptTemplate.fromTemplate(`{{"relationships":[{{"source":"Alice","type":"roommate","target":"Robert"}},{{"source":"Alice","type":"works as","target":"lawyer"}},{{"source":"Robert","type":"roommate","target":"Alice"}},{{"source":"Robert","type":"works as","target":"journalist"}},{{"source":"Alice","type":"owns","target":"alice.com"}},{{"source":"Robert","type":"owns","target":"bob.com"}},{{"source":"Robert","type":"has sister","target":"Alice"}},{{"source":"Alice","type":"has brother","target":"Robert"}}]}}`),
//     HumanMessagePromptTemplate.fromTemplate(`Suggested relationships: ["owns","works as","has sister","has brother","has child","has parent"]
// Entities: ["Robert","shoes","park","David"]
// Novel extract: That morning, Robert put on his shoes, and went to the park.
// David saw him there. "Hi, Bob!", said Dave.`),
//     AIMessagePromptTemplate.fromTemplate(`{{"relationships":[{{"source":"Robert","type":"owns","target":"shows"}},{{"source":"Robert","type":"goes to","target":"park"}},{{"source":"David","type":"goes to","target":"park"}},{{"source":"David","type":"greets","target":"Robert"}}]}}`),
    HumanMessagePromptTemplate.fromTemplate(`Suggested relationships: {labels}
Entities: {entities}
Novel extract: {data}`),
]);

const findEntitiesChain = RunnableSequence.from([
    findEntitiesPrompt,
    mistralLLMJSON,
    new JsonOutputParser(),
]);
const findRelationshipsChain = RunnableSequence.from([
    findRelationshipsPrompt,
    mistralLLMJSON,
    new JsonOutputParser(),
]);

const bar = new cliProgress.SingleBar({ barsize: 80, format: '{bar} {value}/{total} splits | {percentage}% | Time: {duration_formatted} | ETA: {eta_formatted}'}, cliProgress.Presets.shades_classic);
bar.start(splits.length*2, 0);
for(let split of splits) {
    // console.log(chalk.grey(split.pageContent));

    const labels = _.map(await graph.query('CALL db.labels()'), l => l.label);
    let entities = await findEntitiesChain.invoke({
        data: split.pageContent,
        labels: JSON.stringify(_(await graph.query('CALL db.labels()'))
            .map(l => l.label)
            .filter()
            .union([
                'Person',
                'Organisation or group',
                'Location',
                'Job or occupation',
                'Vehicle',
            ])
            .value()),
        entities: JSON.stringify(_(await graph.query('MATCH (n) RETURN n.id as label'))
            .map(l => l.label)
            .filter()
            .value()),
    });

    console.log('\n',chalk.greenBright(JSON.stringify(entities)),'\n');
    bar.increment();

    // We need to look up the original entities to merge properties with it, including finding any array properties
    // and merging those into the existing array values.
    entities = await Promise.all(_(entities.entities)
        .filter('id')
        .filter('type')
        .map(async (e) => {
            // Escape the type name in case it holds any backtick characters
            const type = e.type && e.type.replaceAll('`', '``') || undefined
            // Find the node in the DB if it already exists
            const existing = await graph.query(`MATCH (n:\`${e.type}\`) WHERE n.id = "${e.id}" RETURN n`);
            if(existing && existing[0]) {
                e.properties = _.mergeWith(existing[0].n.properties, e.properties, (objValue, srcValue) => {
                    if(_.isArray(objValue) && _.isArray(srcValue)) {
                        return _.union(objValue, srcValue);
                    }
                });
            }
            return new Node({
                id: e.id,
                type,
                properties: e.properties,
            });
        })
        .value());
    if (entities.length == 0) {
        bar.increment(); // Increment for skipped relation check
        continue;
    }

    let relationships = await findRelationshipsChain.invoke({
        data: split.pageContent,
        entities: JSON.stringify(_(await graph.query('MATCH (n) RETURN n.id as label'))
            .map(l => l.label)
            .filter()
            .union(_.map(entities, e => e.id))
            .value()),
        labels: JSON.stringify(_(await graph.query('CALL db.relationshipTypes()'))
            .map(l => l.label)
            .filter()
            .union([
                'owns',
                'alias of',
                'works as',
                'sister of',
                'brother of',
                'child of',
                'parent of',
                'lives in',
                'located in',
                'hates',
                'likes',
                'in love with',
                'friends with',
            ])
            .value())
    });

    console.log('\n', chalk.greenBright(JSON.stringify(relationships)), '\n');
    bar.increment();

    relationships = _(relationships.relationships)
        .map(r => new Relationship({
            source: _.find(entities, { id: r.source }),
            type: r.type && r.type.replaceAll(' ','_') || undefined,
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
