// TODO: Put novel metadata in the state like title, author name, date, ... so that the LLM has more context and doesn't invent novel names

import { OllamaEmbeddings } from '@langchain/ollama';
import { CacheBackedEmbeddings } from 'langchain/embeddings/cache_backed';
import { InMemoryStore } from 'langchain/storage/in_memory';
import { ChatOllama } from '@langchain/ollama';
import { DynamicStructuredTool } from '@langchain/core/tools';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { JsonOutputToolsParser } from '@langchain/core/output_parsers/openai_tools';
// import { HumanMessage, BaseMessage, AIMessage, ToolMessage } from '@langchain/core/messages';
import { ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate } from '@langchain/core/prompts';
import { END, START, StateGraph } from '@langchain/langgraph';
import { FaissStore } from '@langchain/community/vectorstores/faiss';
import { HydeRetriever } from 'langchain/retrievers/hyde';
import { StringPromptValue } from '@langchain/core/prompt_values';
import { maximalMarginalRelevance } from '@langchain/core/utils/math';

import { z } from 'zod';

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

/* eslint-disable no-unused-vars -- Leave all these so switching is easier without dealing with comments */
const commonOptions = { temperature: 1, seed: 19740822, keepAlive: '15m' };
const commonOptions8k = { numCtx: 8 * 1024, ...commonOptions };
const commonOptions32k = { numCtx: 32 * 1024, ...commonOptions };
const commonOptions64k = { numCtx: 64 * 1024, ...commonOptions };
const commonOptions128k = { numCtx: 128 * 1024, ...commonOptions };
const commonOptions256k = { numCtx: 256 * 1024, ...commonOptions };

const embeddings = CacheBackedEmbeddings.fromBytesStore(
    // new OllamaEmbeddings({ model: 'nomic-embed-text', requestOptions: { numCtx: 2048 } }),
    // new OllamaEmbeddings({ model: 'mxbai-embed-large', requestOptions: { numCtx: 512 } }),
    // new OllamaEmbeddings({ model: 'bge-large', requestOptions: { numCtx: 512 } }),
    new OllamaEmbeddings({ model: 'bge-m3', requestOptions: { numCtx: 8192 } }),
    // new OllamaEmbeddings({ model: 'mistral:7b-instruct-v0.2-q8_0', requestOptions: { numCtx: 32768 } }),
    // new OllamaEmbeddings({ model: 'llama3.1:8b-instruct-q8_0', requestOptions: { numCtx: 2048 } }),
    new InMemoryStore(),
    {
        namespace: 'embeddings',
    }
);

// Prompt parse: ~500-1500 t/s; generation: ~60-70 t/s; HAS TOOLS
const qwen25_3bLLM = new ChatOllama({ model: 'qwen2.5:3b-instruct-q8_0', ...commonOptions32k });

// mistral-nemo has 1024k ctx; HAS TOOLS
// Prompt parse: ~300-500 t/s; generation: ~25-40 t/s
const nemo_12bLLM = new ChatOllama({ model: 'mistral-nemo:12b-instruct-2407-q8_0', ...commonOptions128k });

// llama3.1 has 128k ctx; HAS TOOLS
// Prompt parse: ~200-500 t/s; generation: ~40 t/s
const llama31_8bLLM = new ChatOllama({ model: 'llama3.1:8b-instruct-q8_0', ...commonOptions64k });

// phi3:medium-128k-instruct-q8_0 has 128k ctx but we'll only use 64k; MAYBE NO TOOLS?
// Prompt parse: ~100-250 t/s; generation: ~20 t/s
const phi3_14bLLM = new ChatOllama({ model: 'phi3:14b-medium-128k-instruct-q8_0', ...commonOptions64k });

// mistral-small has 32k training ctx; claims it can do up to 128k ctx; HAS TOOLS
// Prompt parse: ~60-70 t/s; generation: ~15 t/s
const mistralSmallLLM = new ChatOllama({ model: 'mistral-small:22b-instruct-2409-q8_0', ...commonOptions32k });

// qwen2.5 has 128k training ctx; HAS TOOLS
// Prompt parse: ~50-60 t/s; generation: ~10 t/s
const qwen25_32bLLM = new ChatOllama({ model: 'qwen2.5:32b-instruct-q8_0', ...commonOptions32k });

// command-r has 128k ctx; HAS TOOLS
// Prompt parse: ~70 t/s; generation: ~10 t/s
const commandR_35bLLM = new ChatOllama({ model: 'command-r:35b-08-2024-q8_0', ...commonOptions64k });

// llama3.1 has 128k ctx; HAS TOOLS
// Prompt parse: ~30-60 t/s; generation: ~5 t/s
const llama31_70bLLM = new ChatOllama({ model: 'llama3.1:70b-instruct-q8_0', ...commonOptions64k });

// mistral-large has 32k training ctx; claims it can do up to 128k ctx; HAS TOOLS
// Prompt parse: ~12-15 t/s; generation: ~5-6 t/s
const mistralLLM = new ChatOllama({ model: 'mistral-large:latest', ...commonOptions32k });

// qwen2.5 has 128k ctx; HAS TOOLS
// Prompt parse: ~30 t/s; generation: ~5 t/s
const qwen25_72bLLM = new ChatOllama({ model: 'qwen2.5:72b-instruct-q8_0', ...commonOptions32k });

// bespoke-minicheck:7b-q8_0 is a fact checker
// Prompt parse: ~600 t/s; generation: ~50 t/s
const bespokeMinicheckLLM = new ChatOllama({ model: 'bespoke-minicheck:7b-q8_0', ...commonOptions32k });

/* eslint-enable no-unused-vars -- Leave all these so switching is easier without dealing with comments */

const fastLLM = nemo_12bLLM;
const slowLLM = mistralSmallLLM;

const book = 'Christmas Town beta';
const storeDirectory = `novels/${book}`;

/**
 * Return documents selected using the maximal marginal relevance.
 * Maximal marginal relevance optimizes for similarity to the query AND diversity
 * among selected documents.
 *
 * @param {string} query - Text to look up documents similar to.
 * @param {number} options.k - Number of documents to return.
 * @param {number} options.fetchK=20- Number of documents to fetch before passing to the MMR algorithm.
 * @param {number} options.lambda=0.5 - Number between 0 and 1 that determines the degree of diversity among the results,
 *                 where 0 corresponds to maximum diversity and 1 to minimum diversity.
 * @param {MongoDBAtlasFilter} options.filter - Optional Atlas Search operator to pre-filter on document fields
 *                                      or post-filter following the knnBeta search.
 *
 * @returns {Promise<Document[]>} - List of documents selected by maximal marginal relevance.
 */
async function maxMarginalRelevanceSearch(query, options) {
    const { k, fetchK = 20, lambda = 0.5 } = options;
    const queryEmbedding = await this.embeddings.embedQuery(query);
    // preserve the original value of includeEmbeddings
    const includeEmbeddingsFlag = options.filter?.includeEmbeddings || false;
    // update filter to include embeddings, as they will be used in MMR
    const resultDocs = await this.similaritySearchVectorWithScore(queryEmbedding, fetchK);
    const embeddingList = await this.embeddings.embedDocuments(_.map(resultDocs, '0.pageContent'));
    const mmrIndexes = maximalMarginalRelevance(queryEmbedding, embeddingList, lambda, k);
    return _.map(mmrIndexes, (idx) => {
        const doc = resultDocs[idx][0];
        // remove embeddings if they were not requested originally
        if(!includeEmbeddingsFlag) {
            delete doc.metadata[this.embeddingKey];
        }
        return doc;
    });
}

const vectorStore = await FaissStore.load(
    storeDirectory,
    embeddings
);
vectorStore.maxMarginalRelevanceSearch = maxMarginalRelevanceSearch.bind(vectorStore);

const hydePrompt = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(`# Instructions
Write a short paragraph which responds to the query.`),
    HumanMessagePromptTemplate.fromTemplate(`# Query
{query}`),
]);

async function mmrSearch(query, runManager) {
    let value = new StringPromptValue(query);
    // Use a custom template if provided
    if(this.promptTemplate) {
        value = await this.promptTemplate.formatPromptValue({ query });
    }
    // Get a hypothetical answer from the LLM
    const res = await this.llm.generatePrompt([value]);
    const answer = res.generations[0][0].text;
    // Retrieve relevant documents based on the hypothetical answer
    if(this.searchType === 'mmr') {
        if(_.isFunction(this.vectorStore.maxMarginalRelevanceSearch) === false) {
            throw new Error(`The vector store backing this retriever, ${this._vectorstoreType()} does not support max marginal relevance search.`);
        }
        return this.vectorStore.maxMarginalRelevanceSearch(answer, {
            k: this.k,
            filter: this.filter,
            ...this.searchKwargs,
        }, runManager?.getChild('vectorstore'));
    }
    return this.vectorStore.similaritySearch(answer, this.k, this.filter, runManager?.getChild('vectorstore'));
}

const qaRetriever = new HydeRetriever({
    // verbose: true,
    vectorStore,
    llm: fastLLM, // Basic task to write the prompt so do it quickly
    searchType: 'mmr',
    searchKwargs: {
        lambda: 0.5,
        fetchK: 100,
    },
    k: 10,
    promptTemplate: hydePrompt,
});
qaRetriever._getRelevantDocuments = mmrSearch.bind(qaRetriever);

const sortDocsFormatAsJSON = (documents) => {
    return JSON.stringify(
        _(documents)
            .sortBy(['metadata.source', 'metadata.loc.pageNumber', 'metadata.loc.lines.from'])
            .map(doc => ({
                loc: doc.metadata.loc,
                text: doc.pageContent,
            }))
            .value()
    );
};

const graphState = {
    documents: {
        value: (left, right) => right ?? left ?? [],
        'default': () => [],
    },
    filteredDocuments: {
        value: (left, right) => (right ? _.uniqBy([...left, ...right], 'pageContent') : left ?? []),
        'default': () => [],
    },
    uselessDocuments: {
        value: (left, right) => (right ? _.uniqBy([...left, ...right], 'pageContent') : left ?? []),
        'default': () => [],
    },
    origQuery: {
        value: (left, right) => right ?? left ?? '',
        'default': _.constant(''),
    },
    priorQueries: {
        value: (left, right) => (right ? [...left, ...right] : left ?? []),
        'default': () => [],
    },
    query: {
        value: (left, right) => right ?? left,
        'default': () => undefined,
    },
    generation: {
        value: (left, right) => right ?? left,
        'default': () => undefined,
    }
};

/**
 * Call the retriever to find matching documents
 * @param {GraphState} state - The current state of the agent, including the query.
 * @returns {Promise<GraphState>} - The updated state with the documents added.
 */
async function retrieve(state) {
    logger.debug('---EXECUTE RETRIEVAL---');

    // We call the tool_executor and get back a response.
    logger.debug(chalk.greenBright(JSON.stringify(state.query || state.origQuery)));
    const documents = await qaRetriever
        .withConfig({ runName: 'FetchRelevantDocuments' })
        .invoke(state.query || state.origQuery);

    return { documents, query: state.query || state.origQuery };
}

// eslint-disable-next-line no-unused-vars -- Keep this definition as an alternative to gradeDocuments to keep all
async function passthroughAllDocuments(state) {
    return { filteredDocuments: state.documents, uselessDocuments: [], documents: [] };
}

/**
 * Determines whether the retrieved documents are relevant to the query. Filters out documents which have already been added to the list of relevant ones.
 * @param {GraphState} state - The current state of the graph, including query and documents.
 * @returns {Promise<GraphState>} - The updated state with documents filtered for relevance.
 */
async function gradeDocuments(state) {
    logger.debug('---GET RELEVANCE---');
    // Output
    const tool = new DynamicStructuredTool({
        name: 'give_relevance_score',
        description: 'Give a relevance score to the retrieved documents.',
        schema: z.object({
            relevanceScore: z.string().describe("Score 'yes' or 'no'"),
        }),
        func: async ({ relevanceScore }) => relevanceScore,
    });

    const prompt = ChatPromptTemplate.fromMessages([
        SystemMessagePromptTemplate.fromTemplate(`# Preamble
You are a grader assessing the relevance of a short extract from a novel to a user's query about the novel. The extract will be combined with other extracts and provided to another LLM in order to answer a query.
You are *not* answering the query, you are merely assessing the relevance of the extract to the query.

# Instructions
Give a relevance score 'yes' or 'no' score to indicate whether the extract is in any way relevant to the query. Err on the side of saying that a document is relevant, if you're not really sure.
yes: The extract is at least vaguely relevant to the query.
no: The extract is not at all relevant to the query.`),
        HumanMessagePromptTemplate.fromTemplate(`# Extract
\`\`\`
{extract}
\`\`\`

# Query
{query}
`)]);

    const llm = fastLLM;
    const oldTemp = llm.temperature;
    llm.temperature = 0;

    const model = llm.bindTools([tool]);
    const chain = prompt.pipe(model).pipe(new JsonOutputToolsParser());

    logger.debug(`${state.documents.length} orig docs`);
    const reducedDocs = _(state.documents)
                        .filter(d => !_.some(state.filteredDocuments, { pageContent: d.pageContent }))
                        .filter(d => !_.some(state.uselessDocuments, { pageContent: d.pageContent }))
                        .value();
    logger.debug(`${reducedDocs.length} reduced docs`);

    const filteredDocuments = [];
    const uselessDocuments = [];
    for await (const doc of reducedDocs) {
        const grade = await chain.invoke({
            extract: doc.pageContent,
            query: state.origQuery,
        });
        // logger.debug(grade);
        if(grade?.[0]?.args?.relevanceScore === 'yes') {
            logger.debug('---GRADE: DOCUMENT RELEVANT---');
            filteredDocuments.push(doc);
        } else {
            logger.debug('---GRADE: DOCUMENT NOT RELEVANT---');
            uselessDocuments.push(doc);
        }
    }

    llm.temperature = oldTemp;
    return { documents: [], filteredDocuments, uselessDocuments };
}

/**
 * Transform the query to produce a better query.
 *
 * @param {GraphState} state The current state of the graph.
 * @returns {Promise<GraphState>} The new state object.
 */
async function transformQuery(state) {
    logger.debug(`---TRANSFORM QUERY: ${state.priorQueries.length} PREVIOUS QUERIES---`);

    const prompt = ChatPromptTemplate.fromMessages([
        SystemMessagePromptTemplate.fromTemplate(`# Background
You are generating a query that is well optimized for semantic search retrieval.

# Instructions
Look at the initial query, and previous attempts at re-writing the query and try to reason about the underlying semantic intent / meaning. Then formulate and reply with an improved query more likely to surface responsive documents. Do not *answer* the query, just re-write it in a way that is more likely to get a good answer.

# Output format
Your output should be just the re-written query, with no discussion, pre-amble, formatting, or other considerations, just the text of the improved query.`),
        HumanMessagePromptTemplate.fromTemplate(`# Previous inadequate queries
{previous_queries}

# Initial query
{query}
`)]);

    // Prompt
    const oldTemp = fastLLM.temperature;
    const oldCtx = fastLLM.numCtx;
    fastLLM.temperature = 2;
    fastLLM.numCtx = 4096;
    const chain = prompt.pipe(fastLLM).pipe(new StringOutputParser());
    const betterQuery = await chain.invoke({
        query: state.origQuery,
        previous_queries: state.priorQueries.join('\n'),
    });
    fastLLM.temperature = oldTemp;
    fastLLM.numCtx = oldCtx;

    return {
        query: betterQuery,
        priorQueries: [state.query],
    };
}

/**
 * Determines whether to generate an answer, or re-generate a question.
 *
 * @param {GraphState} state The current state of the graph.
 * @returns {"transformQuery" | "generate"} Next node to call
 */
function decideToGenerate(state) {
    logger.debug(`---DECIDE TO GENERATE: ${state.filteredDocuments.length} RELEVANT DOCUMENTS---`);
    const filteredDocuments = state.filteredDocuments;

    if(filteredDocuments.length <= 30 && state.priorQueries.length < 5) {
        //
        // Too many documents have been filtered checkRelevance
        // We will re-generate a new query
        logger.debug('---DECISION: TRANSFORM QUERY---');
        return 'transformQuery';
    }
    // We have relevant documents, so generate answer
    logger.debug('---DECISION: GENERATE---');
    return 'generate';
}

const mainAgentPromptTemplate = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(`# Basic instructions
You are a powerful conversational AI trained to work as a developmental editor, assisting authors (as users) to improve their unpublished novels before the drafts are submitted to literary agents to find a publisher. You are provided with one or more extracts from the novel which should contain the answers to a query from the user, and your job is to digest these extracts to best help the user. When you answer the user's requests, you cite your sources in your answers.

## Task and context
You help authors answer their questions and other requests. You will be asked a very wide array of requests on all kinds of topics. You should use the provided extract(s) to answer the question, and not make anything up that wasn't in at least one of the extracts. You should focus on serving the user's needs as best you can.
Your job is to help find the book's flaws when they exist, and suggest to the author how they might fix them - that is the whole point of your review. Analyze any flaws rigorously and do not just mindlessly praise the author's work.

## Limitations
Remember that you're only reading a few extracts from the novel. You can get some sense of how much you're not seeing based on the provided location data which tells you which lines or pages of the book each extract is from. You will see that you're only seeing a very limited chunk of the novel.

## Style guide
Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling. Use Markdown to improve the formatting and presentation of your final answer.
`),
    HumanMessagePromptTemplate.fromTemplate(`# Extracts
\`\`\`json
{extracts}
\`\`\`

# Query
{query}
`)]);

/**
 * Generate answer
 *
 * @param {GraphState} state The current state of the graph.
 * @param {RunnableConfig | undefined} config The configuration object for tracing.
 * @returns {Promise<GraphState>} The new state object.
 */
async function generate(state) {
    logger.debug(`---GENERATE FROM ${state.filteredDocuments.length} DOCS---`);
    // Pull in the prompt
    const prompt = mainAgentPromptTemplate;

    // LLM
    const llm = slowLLM;

    // RAG Chain
    const ragChain = prompt.pipe(llm).pipe(new StringOutputParser());

    const docs = sortDocsFormatAsJSON(state.filteredDocuments);
    logger.debug(`Context has length ${docs.length}`);

    const generation = await ragChain.invoke({
        extracts: sortDocsFormatAsJSON(state.filteredDocuments),
        query: state.origQuery,
    });

    return {
        filteredDocuments: [],
        generation,
    };
}

const workflow = new StateGraph({ channels: graphState })
    .addNode('retrieve', retrieve)
    .addNode('gradeDocuments', gradeDocuments)
    // .addNode('gradeDocuments', passthroughAllDocuments)
    .addNode('transformQuery', transformQuery)
    .addNode('generate', generate)
    .addEdge(START, 'retrieve')
    .addEdge('retrieve', 'gradeDocuments')
    .addConditionalEdges('gradeDocuments', decideToGenerate)
    .addEdge('transformQuery', 'retrieve')
    .addEdge('generate', END);

const app = workflow.compile();

// eslint-disable-next-line @stylistic/operator-linebreak -- This is fine here cos we can swap in any of the prompts
const input =
    // `What do you think of this novel?`
    // `Identify sentences which are too long or complex and are hard to understand.`
    // `What would be a good, engaging title for this novel?`
    // `Give a precis of the novel: list genre, describe the protagonist and major characters, and provide an overall plot summary.`
    // `Analyze the story, and let me know if you think this is similar to any other well-known stories in its genre, or in another genre.`
    // `Proofreading: are there any spelling, grammar, or punctuation errors that can distract readers from the story itself? Please list them all, including reference information for where they occur in the novel.`
    // `Character development: Identify the important characters and then assess how well-developed they are, with distinct personalities, backgrounds, and motivations.`
    // `Plot structure: Analyze whether the story's events are in a clear and coherent sequence, with rising action, climax, falling action, and resolution.`
    // `Subplots: Analyze the sub-plots and minor characters to verify that they add to the story instead of distracting from it. Sub-plots and side-characters should enhance the story and not confuse the reader. Point out any flaws.`
    // `Show, don't tell: Analyze whether the story simply tells readers what is happening or how characters feel, or whether it uses vivid descriptions and actions to show them. This will make the writing more engaging and immersive.`
    // `Consistent point of view: Does the novel stick to one consistent point of view throughout, whether it be first person, third person limited, or omniscient? This will help maintain a cohesive narrative voice.`
    // `Active voice: Does the writing use active voice instead of passive voice whenever possible?`
    // `Vary sentence structure: Does the writing break up long sentences with shorter ones to create rhythm and variety?`
    // `Analyze the story from the point of view of a potential reader who purchases the book. Would they be likely to enjoy reading it?`
    // `Analyze the story from the point of view of a literary agent reading this book for the first time and trying to decide if they want to represent this author to publishers.`
    // `Provide suggestions on how to improve any confusing parts of the plot. If there are other narrative elements which should be revised and improved, point them out.`
    // `List all the chapters in the book, and give a one-sentence summary of each chapter.`
    // `What do you dislike the most about the book? What needs fixing most urgently?`
    // `Who is the ideal audience for this book? What will they enjoy about it? What might they dislike about it? How can the story be adjusted to make it appear to a wider audience?`
    // `Identify any repetitive or superfluous elements in the book.`
    // `Identify any subplots which don't lead anywhere and just distract from the main story.`
    // `Write a dust-jacket blurb describing this novel and a plot synopsis, in an engaging way but without spoilers, with some invented (but realistic) quotes from reviewers about how good the book is. One of the reviewers should be "OpenAI ChatGPT". Do not include any extracts from the book itself. Use markdown syntax for formatting.`
    // `Write a detailed query letter to a potential literary agent, explaining the novel and how it would appeal to readers. The letter should be engaging, and should make the agent interested in representing the book, without being overly cloying or sounding desperate. Be sure to properly research the book content so you're not being misleading. Find out the names of any characters mentioned. The agent will not have read the novel, so any discussion of the novel should not assume that the agent has read it yet. Reference the major events that happen in the book, describe what makes the protagonist engaging for readers, and include something about why the author chose to write this story.`
    // `Is Meghan a likable and relatable character for readers, even if characters in the book perhaps dislike her? Will readers be able to empathize with her and enjoy the novel with her as the protagonist?`
    // `Does Bathrobe Grouch have a real name?`
    // `What is Mr Zimmerman's nickname?`
    `How does it turn out that Tyler died in the end? Who is responsible?`
    // `What is the age of the main character and what are some of the challenges she faces throughout the novel?`
    // `How can the story be adjusted to make it appealing to a wider audience without losing its core themes of trauma, loss, and redemption?`
    // `Are there any secondary characters or subplots in the novel that could be expanded upon to provide additional perspectives or interests?`
    // `Can you provide more context about Roger and his role in the novel? How does he relate to the themes of family, loss, and personal growth?`
    // `Can you provide a brief overview of the main plot points? Is the story believable?`
    // `Pick any quotation from the book and count the number of words in it.`
    // `Should Tyler's body be found earlier in the narrative? I'm not talking about figuring out how he died, just the actual discovery of his death. Typically, this discovery would be the inciting incident in a mystery novel but this isn't purely a mystery novel. Have I been successful in engaging readers in Meghan's life so that postponing the mystery elements of the novel works?`
    // `Meghan at times uses obscure words; is it unbelievable that a highschool sophomore would know these words, given Meghan's character and background?`
    // `Are there any instances in the novel where I've misused words? That is, where the word is used in a way that is not consistent with the meaning of the word?`
    // `Write a 1000-word comparative literature essay about this novel written in 2024, thinking about it as an allegory for the modern world of AI, even though the story is set in the 1990s before such AI had been developed. How do the novel's themes mirror issues of social isolation in a world full of robots? How does it help us understand how human societies can co-exist with robots while maintaining any notion of a human "self"? Do not invent things which are not in the novel itself; use quotes from the novel as appropriate to support your arguments. Draw on external sources as necessary. Use markdown formatting in the essay, including markdown footnotes for bibliographic references.`
    // `Pick an iconic scene from the book, and describe it in visual detail. The description will be provided to an AI image generator using a Stable Diffusion type model. Include in your description all the important elements which will allow the AI model to properly generate the image. The image generator knows nothing of the novel, so if it's important, include things like the period/era of the story, the geographical setting, etc. so an accurate image can be generated. Do not include any preamble, discussion or any other meta-information, merely output the description of the desired image.`
    // `When is this story set? What decade, or if you can be more specific, what year? How can you tell? Are there any clues in pop culture references in the story like TV shows, movies, songs, books, or anything similar?`
    ;

const inputs = {
    origQuery: input,
};
const config = { recursionLimit: 50, streamMode: 'values' };
let finalState;
for await (const output of await app.stream(inputs, config)) {
    if(!output.generation) {
        logger.info(output.filteredDocuments);
    } else {
        finalState = output;
    }
}

logger.info(chalk.whiteBright(finalState.generation));
