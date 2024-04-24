import chalk from 'chalk';
import _ from 'lodash';

import { OllamaEmbeddings } from '@langchain/community/embeddings/ollama';
import { ChatOllama } from '@langchain/community/chat_models/ollama';
import { FaissStore } from '@langchain/community/vectorstores/faiss';
import { HydeRetriever } from 'langchain/retrievers/hyde';
import { StringPromptValue, } from '@langchain/core/prompt_values';
import { maximalMarginalRelevance } from '@langchain/core/utils/math';

import { RunnableSequence, RunnablePassthrough } from '@langchain/core/runnables';
import { DynamicTool } from '@langchain/core/tools';
import { WikipediaQueryRun } from '@langchain/community/tools/wikipedia_query_run';
import { DuckDuckGoSearch } from '@langchain/community/tools/duckduckgo_search';

import { AgentExecutor } from 'langchain/agents';

import { ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate } from '@langchain/core/prompts';

import { renderTextDescription } from 'langchain/tools/render';
import { ReActSingleInputOutputParser } from './node_modules/langchain/dist/agents/react/output_parser.js';
import { RunnableSingleActionAgent } from './node_modules/langchain/dist/agents/agent.js';
import { StringOutputParser, JsonOutputParser } from 'langchain/schema/output_parser';

function formatLogToString(intermediateSteps, observationPrefix = "Observation: ", llmPrefix = "Thought: ") {
    const formattedSteps = intermediateSteps.reduce((thoughts, { action, observation }) => thoughts +
        [action.log, `\n${observationPrefix}${observation}`, llmPrefix].join("\n"), "");
    return formattedSteps;
}

// Re-writing this so that it doesn't do the scratchpad wrong for instruct model
async function createReactAgent({ llm, tools, prompt, streamRunnable, }) {
    const missingVariables = ["tools", "tool_names", "agent_scratchpad"].filter((v) => !prompt.inputVariables.includes(v));
    if (missingVariables.length > 0) {
        throw new Error(`Provided prompt is missing required input variables: ${JSON.stringify(missingVariables)}`);
    }
    const toolNames = tools.map((tool) => tool.name);
    const partialedPrompt = await prompt.partial({
        tools: renderTextDescription(tools),
        tool_names: toolNames.join(", "),
    });
    // TODO: Add .bind to core runnable interface.
    const llmWithStop = llm.bind({
        stop: ["\nObservation:"],
    });
    const agent = RunnableSequence.from([
        RunnablePassthrough.assign({
            agent_scratchpad: (input) => formatLogToString(input.steps, undefined, ''),
        }),
        partialedPrompt,
        llmWithStop,
        new ReActSingleInputOutputParser({
            toolNames,
        }),
    ]);
    return new RunnableSingleActionAgent({
        runnable: agent,
        defaultRunName: "ReactAgent",
        streamRunnable,
    });
}

// const embeddings = new OllamaEmbeddings({ model: 'nomic-embed-text', numCtx: 2048, baseUrl: 'http://127.0.0.1:11434' });
const embeddings = new OllamaEmbeddings({ model: 'mistral:instruct', numCtx: 32768 });

const commonOptions = { temperature: 0, seed: 19740822 };
const commonOptionsJSON = { format: 'json' };
const commonOptions8k = { numCtx: 8 * 1024, ...commonOptions };
const commonOptions8kJSON = { ...commonOptions8k, ...commonOptionsJSON };
const commonOptions32k = { numCtx: 32 * 1024, ...commonOptions };
const commonOptions32kJSON = { ...commonOptions32k, ...commonOptionsJSON };
const commonOptions64k = { numCtx: 64 * 1024, ...commonOptions };
const commonOptions64kJSON = { ...commonOptions64k, ...commonOptionsJSON };

// mistral7b-instruct has 32k training ctx but ollama sets it to 2k so need to override that here
// Prompt parse: ~550-600 t/s; generation: ~50-60 t/s
const mistralLLMChat = new ChatOllama({ model: 'mistral:instruct', ...commonOptions32k, baseUrl: 'http://127.0.0.1:11435' });
const mistralLLMJSON = new ChatOllama({ model: 'mistral:instruct', ...commonOptions32kJSON, baseUrl: 'http://127.0.0.1:11435' });

// wizardlm2:7b-q5_1 has 32k training ctx but ollama sets it to 2k so need to override that here, it's quite bad at long context though.
// Prompt parse: 500-650 t/s; generation: ~40-50 t/s
const wizard7bLLMChat = new ChatOllama({ model: 'wizardlm2:7b-q5_1', ...commonOptions8k, baseUrl: 'http://127.0.0.1:11435' });
const wizard7bLLMJSON = new ChatOllama({ model: 'wizardlm2:7b-q5_1', ...commonOptions8kJSON, baseUrl: 'http://127.0.0.1:11435' });

// llama3 llama3:8b-instruct-q5_K_M has 8k ctx
const llama3LLMChat = new ChatOllama({ model: 'llama3:8b-instruct-q5_K_M', ...commonOptions8k, baseUrl: 'http://127.0.0.1:11435' });
const llama3LLMJSON = new ChatOllama({ model: 'llama3:8b-instruct-q5_K_M', ...commonOptions8kJSON, baseUrl: 'http://127.0.0.1:11435' });

// mixtral:8x7b-instruct-v0.1-q5_K_M - 32k context
// Prompt parse: ~150-200 t/s; generation: ~20-25 t/s
const mixtral7BLLMChat = new ChatOllama({ model: 'mixtral:8x7b-instruct-v0.1-q5_K_M', ...commonOptions32k, baseUrl: 'http://127.0.0.1:11436' });
const mixtral7BLLMJSON = new ChatOllama({ model: 'mixtral:8x7b-instruct-v0.1-q5_K_M', ...commonOptions32kJSON, baseUrl: 'http://127.0.0.1:11436' });

// dolphin-mixtral:8x7b-v2.7-q6_K - 32k training context but ollama sets it to 2k
// Prompt parse: ~150-200 t/s; generation: ~20-25 t/s
const dolphmix7BLLMChat = new ChatOllama({ model: 'dolphin-mixtral:8x7b-v2.7-q6_K', ...commonOptions32k, baseUrl: 'http://127.0.0.1:11436' });
const dolphmix7BLLMJSON = new ChatOllama({ model: 'dolphin-mixtral:8x7b-v2.7-q6_K', ...commonOptions32kJSON, baseUrl: 'http://127.0.0.1:11436' });

// command-r-plus:104b-q4_0 has 128k training ctx but ollama sets it to 2k so need to override that here
// Prompt parse: ~50-60t/s; generation: ~5-6 t/s
const commandRLLMChat = new ChatOllama({ model: 'command-r-plus:104b-q4_0', ...commonOptions32k, baseUrl: 'http://127.0.0.1:11437' });
const commandRLLMJSON = new ChatOllama({ model: 'command-r-plus:104b-q4_0', ...commonOptions32kJSON, baseUrl: 'http://127.0.0.1:11437' });

// mixtral:8x22b-instruct-v0.1-q4_0 - 64k training context but ollama sets it to 2k, has special tokens for tools and shit
// Prompt parse: ~60-80 t/s; generation ~11-12 t/s
const mixtral22bLLMChat = new ChatOllama({ model: 'mixtral:8x22b-instruct-v0.1-q4_0', ...commonOptions64k, baseUrl: 'http://127.0.0.1:11437' });
const mixtral22bLLMJSON = new ChatOllama({ model: 'mixtral:8x22b-instruct-v0.1-q4_0', ...commonOptions64kJSON, baseUrl: 'http://127.0.0.1:11437' });

const storeDirectory = 'novels/Christmas Town draft 2';
// const storeDirectory = 'novels/Fighters_pages';

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
    const { k, fetchK = 20, lambda = 0.5, filter } = options;
    const queryEmbedding = await this.embeddings.embedQuery(query);
    // preserve the original value of includeEmbeddings
    const includeEmbeddingsFlag = options.filter?.includeEmbeddings || false;
    // update filter to include embeddings, as they will be used in MMR
    const resultDocs = await this.similaritySearchVectorWithScore(queryEmbedding, fetchK);
    const embeddingList = await Promise.all(resultDocs.map((doc) => this.embeddings.embedQuery(doc[0].pageContent)));
    const mmrIndexes = maximalMarginalRelevance(queryEmbedding, embeddingList, lambda, k);
    return mmrIndexes.map((idx) => {
        const doc = resultDocs[idx][0];
        // remove embeddings if they were not requested originally
        if (!includeEmbeddingsFlag) {
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
Provide 3 alternate phrasings of the query (with no formatting, bullets or numbering), and then write a short paragraph which responds to the query.`),
    HumanMessagePromptTemplate.fromTemplate(`Query: What's your name?`),
    AIMessagePromptTemplate.fromTemplate(`
What do you call yourself?
What is your name?
How do you say your name?

My name is Simon Smith`),
    HumanMessagePromptTemplate.fromTemplate('Query: {question}'),
]);

async function mmrSearch(query, runManager)
{
    let value = new StringPromptValue(query);
    // Use a custom template if provided
    if (this.promptTemplate) {
        value = await this.promptTemplate.formatPromptValue({ question: query });
    }
    // Get a hypothetical answer from the LLM
    const res = await this.llm.generatePrompt([value]);
    const answer = res.generations[0][0].text;
    // Retrieve relevant documents based on the hypothetical answer
    if (this.searchType === "mmr") {
        if (typeof this.vectorStore.maxMarginalRelevanceSearch !== "function") {
            throw new Error(`The vector store backing this retriever, ${this._vectorstoreType()} does not support max marginal relevance search.`);
        }
        return this.vectorStore.maxMarginalRelevanceSearch(answer, {
            k: this.k,
            filter: this.filter,
            ...this.searchKwargs,
        }, runManager?.getChild("vectorstore"));
    }
    return this.vectorStore.similaritySearch(answer, this.k, this.filter, runManager?.getChild("vectorstore"));
}

// const retriever = vectorStore.asRetriever({ k: 15 });
const qaRetriever = new HydeRetriever({
    // verbose: true,
    vectorStore,
    llm: mistralLLMChat, // Basic task to write the prompt so do it quickly
    searchType: 'mmr',
    searchKwargs: {
        lambda: 0.75,
        fetchK: 100,
    },
    k: 50,
    promptTemplate: hydePrompt,
});
qaRetriever._getRelevantDocuments = mmrSearch.bind(qaRetriever);

const extractRetriever = new HydeRetriever({
    // verbose: true,
    vectorStore,
    llm: mistralLLMChat, // Basic task to write the prompt so do it quickly
    searchType: 'mmr',
    searchKwargs: {
        lambda: 0.75,
        fetchK: 100,
    },
    k: 50,
    promptTemplate: hydePrompt,
});
extractRetriever._getRelevantDocuments = mmrSearch.bind(extractRetriever);

const mainAgentPromptTemplate = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(
`# System Preamble
## Basic Rules
You are a powerful conversational AI trained to work as a developmental editor, assisting authors (as users) to improve their unpublished novels before the drafts are submitted to literary agents to find a publisher. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a query from the user, followed by a history of your thoughts, actions you have taken to date, and the results of those actions (observations). When you answer the user's requests, you cite your sources in your answers.

# User Preamble
## Task and Context
You help authors answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user's needs as best you can, which will be wide-ranging.
Your job is to help find the book's flaws when they exist, and suggest to the author how they might fix them - that is the whole point of your review. Analyze any flaws rigorously and do not just mindlessly praise the author's work.

## Style Guide
Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling. Use Markdown to improve the formatting and presentation of your final answer.

## Available Tools
Here is a list of tools that you have available to you:

{tools}

# How to proceed with your work

1. You will first come up with an overall plan for answering the query, and then break that plan down into simple individual steps, and then iterate through the steps to produce a result, one step at a time.

Every response will first begin with a thought for how to achieve this step, and you will output it like this:

\`\`\`
Thought: you should always think about what additional research might help better answer the question
\`\`\`

2. After providing the thought, you will decide either to use a tool to gather more information, or that you have enough information already for a final answer.

## If you need to use a tool
Tools are stateless, and they do not themselves have access to tools, they only know about the action input that you send to them for that specific request.
Tools also are terrible at answering multi-part questions. So ask one simple question, get the response, then ask another question, instead of combining questions together in one query.
You should generally not use tools to do creative work, suggest solutions, or do complex analysis. Do that work yourself. Only use tools to gather information about the novel, its characters, plot, scenes and so forth, or to looks things up on the internet.
Do not ask compound questions. Ask one simple thing at a time, but ask as many separate questions as you like successively in subsequent tools calls.

You use a tool by replying like this:

\`\`\`
Action: only the name of the tool to use (should be one of [{tool_names}] verbatim with no escaping characters, omit this "Action" line completely if not requesting an Action)
Action Input: the input to the tool (omit this "Action Input" line completely if not requesting an Action, but ALWAYS include it if you do request an Action)
\`\`\`

3. The tool will the produce an Observation in response, like this:

\`\`\`
Observation: the result from the tool
\`\`\`

4. ... (you can then repeat this Thought/Action/Action Input/Observation until you have enough information). Cycle through Though/Action/Observation as many times as necessary - do not take initial tool answers as being exhaustive or conclusive; tools often miss things the first time you ask.

5. When you are all done and have pursued every thought, and you are not requesting another tools use you may move on to a conclusion as a final answer, but do not do this prematurely - make sure you really thought everything through. When you have a final answer, you should use this format:

\`\`\`
Final Answer: the final answer to the original query
\`\`\`

## Remember: Thought (required), [Action/Action Input, Observation] (optional) many times, then Final Answer (at the very end)
Always in your responses then, you should include a single Thought, and either an Action *or* a Final Answer but not both - it's one or the other, but ALWAYS include one of the two.`),
    HumanMessagePromptTemplate.fromTemplate(`Query: Some question about the novel`),
    AIMessagePromptTemplate.fromTemplate(`Thought: An overall plan, broken down into steps. You will then iterate through the steps one by one to compile a final answer.
Action: some-tool
Action Input: Tool query which will produce information about the first step in the plan.`),
    AIMessagePromptTemplate.fromTemplate(`Observation: results from the tool`),
    AIMessagePromptTemplate.fromTemplate(`... (more thought/action/observations)`),
    AIMessagePromptTemplate.fromTemplate(`Thought: Continue executing the plan but amending it as necessary as informed by the observations
Action: more-tool-use
Action Input: input for the tool`),
    AIMessagePromptTemplate.fromTemplate(`Observation: tool results`),
    AIMessagePromptTemplate.fromTemplate(`Thought: Ok, from all the observations so far, I'm ready to produce a final result
Final Answer: The best answer to the query based on the thoughts and observations above.`),
    HumanMessagePromptTemplate.fromTemplate(`Query: {input}`),
    AIMessagePromptTemplate.fromTemplate(`{agent_scratchpad}`),
]);

const qaStuffPromptTemplate = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(`# System Preamble
## Basic Rules
You are a powerful conversational AI trained to work as an assistant to a developmental editor (as user), who in turn is assisting authors to improve their unpublished novels. When you answer the user's requests, you cite your sources in your answers.

# User Preamble
## Task and Context
You help developmental editors answer their questrions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You should focus on serving the user's needs as best you can, which will be wide-ranging.
You are given a limited number of extracts from a semantic index of the novel, in JSON as reference material. Answer the query as completely and accurately as possible by considering these extracts. You should ANSWER THE QUERY, and not just regurgitate verbatim quotes.


## Style Guide
Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling, wrapped in JSON as described below.
Let the user know that they can ask you to read further if necessary to confirm things.

Provide it back to the user in JSON format like this:
\`\`\`
{{"answer":"The sky is blue"}}
\`\`\`

If you don't know the answer, or the answer cannot be found in the extracts, or if you are unsure of the answer for any reason, just say that you don't know for sure, and include your best guess.
You won't get in trouble for saying you don't know. You can do that by responding like this for example:

\`\`\`
{{"answer":"As far as I can tell, the hero dies at the end.","warning":"I can only make an educated guess based on the few short passages that I read."}}
\`\`\`

If the question is vague or multipartite, then ask for it to be broken down into simpler individual questions or to be rephrased in a more precise way.
You can do that like this for example:

\`\`\`
{{"warning":"That question is too complex for me. Could you break it down into simpler questions, and ask them one at a time?"}}
\`\`\``),
    HumanMessagePromptTemplate.fromTemplate(`Editor's query: {question}
Extracts:
{context}`),
]);

const extractStuffPromptTemplate = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(`You provide verbatim extracts or passages from a novel for a developmental editor.
You are given extracts from a semantic index of the novel in JSON.
Given these extracts and a query, return a single extract from the novel that is the most relevant to the query.

Make it clear this is only a single extract, and there might be other extracts that are relevant to the query.
Let your boss know that they can ask for other extracts if they want more.

If there are no relevant extracts, just say so. Don't try to make up an answer.
You won't get in trouble for saying you don't know, and the correct answer in that case is that you do not know.
You can do that by responding like this (omitting any extract), for example:

\`\`\`
{{"warning":"I could not find any relevant extract, sorry! Perhaps try re-phrasing the question?"}}
\`\`\`

If the question is too vague or complex, then ask for it to be broken down into simpler questions or to be more precise - dont give a half-assed answer.
You can do that by responding like this (omitting any extract) for example:

\`\`\`
{{"warning":"The question is vague, so I'm not sure how to answer it. Please could you be more precise in what you'd like me to find?"}}
\`\`\`

If the extract you choose does not cover the entire question, then let your boss know that it's only partly responsive, and that for further extracts they should rephrase the question to focus on parts which weren't covered by the extract your provided. Do include the most suitable extract though, for example:

\`\`\`
{{"extract":{{"text":"Inside, the glass walls reflected not just the world outside but also the soul within, as if inviting visitors to see beyond the veil of reality and touch the very essence of their own being.","loc":{{"lines":{{"from":423,"to":551}}}}}},"commentary":"This extract is a perfect example of symbolism used in the text, which is part of what you were asking for, but doesn't cover any major plot elements, which you had also wanted.","warning":"No single extract really covers everything you asked about. If you want another extract, focus your question more narrowly."}}
\`\`\`

When you find an ideal extract, omit any warning, but you should include your own commentary on the extract, along with the verbatim text.
The commentary should clarify the context of the extract, and identify any pronouns used.
All responses *must* be in syntactically valid JSON for example:

\`\`\`
{{"extract":{{"text":"And then, he stabbed the dragon straight through the heart and killed it.","loc":{{"lines":{{"from":123,"to":456}}}}}},"commentary":"I chose this extract because it shows clearly how the hero defeated the dragon. \\"He\\" in the extract refers to the hero."}}
\`\`\`

When there is an appropriate extract, always return it as demonstrated above. Do not forget to include the actual extract from your response when there is one, and always use the most representative extract that best responds to the query.`),
    HumanMessagePromptTemplate.fromTemplate(`Editor's query: {question}
Extracts to choose from:
{context}`),
]);

const sortDocsFormatAsJSON = (documents) => {
    return JSON.stringify(
        _.chain(documents)
        .sortBy(['metadata.source', 'metadata.loc.pageNumber', 'metadata.loc.lines.from'])
        .map((doc) => ({
            loc: doc.metadata.loc,
            text: doc.pageContent,
        }))
        .value()
    );
};

const qaChain = RunnableSequence.from([
    {
        context: qaRetriever.pipe(sortDocsFormatAsJSON),
        question: new RunnablePassthrough(),
    },

    qaStuffPromptTemplate,

    fastestLLMJSON,

    new JsonOutputParser(),
]);

const extractRetrievalChain = RunnableSequence.from([
    {
        context: extractRetriever.pipe(sortDocsFormatAsJSON),
        question: new RunnablePassthrough(),
    },

    extractStuffPromptTemplate,

    fastestLLMJSON,

    new JsonOutputParser(),
]);

const tools = [
    new DynamicTool({
        // verbose: true,
        name: 'novel-analyst',
        description: 'A tool powered by a Large Language Model that can answer basic questions about story elements of the novel: plot, characters, scenes and such. Tool input should be in complete sentences.',
        func: async (x) => JSON.stringify(await qaChain.invoke(x)),
    }),
    new DynamicTool({
        // verbose: true,
        name: 'quote-extractor',
        description: 'A tool powered by a Large Language Model that provides a single short extract up to at most a paragraph or so long. It is best for very specific verbatim extracts about specific things that happen in the novel. Tool input should be in complete sentences.',
        func: async (x) => JSON.stringify(await extractRetrievalChain.invoke(x)),
    }),
    new DynamicTool({
        name: 'word-count',
        description: 'Count the number of words in the literal input. If you need to count words, send the text to this tool for an accurate number. This tool will treat the input itself as the thing to count, it will not look anything up or interpret the input in any way.',
        func: async (x) => {
            return `That input text had ${_(x).split(/\b/).filter(w => /^\w+$/.test(w)).size()} words`
        },
    }),
    // new WikipediaQueryRun({
    //     topKResults: 3,
    //     maxDocContentLength: 4000,
    // }),
    // new DuckDuckGoSearch({
    //     maxResults: 3
    // }),
    // new ExaSearchResults({
    //     client: new Exa('7980b8df-2d60-4900-bc29-4d3695eb4e45'),
    // }),
];

const agent = await createReactAgent({
    llm: commandRLLMChat,
    tools,
    prompt: mainAgentPromptTemplate,
 });
const executor = new AgentExecutor({
    agent,
    tools,
    // returnIntermediateSteps: true,
    handleParsingErrors: 'Please try again, paying close attention to the output format',
});

const internLookup = {};

const input = // `What do you think of this novel?`
    // `Identify sentences which are too long or complex and are hard to understand.`
    // `What would be a good, engaging title for this novel?`
    // `Who are the main characters in the novel?`
    // `Does Bathrobe Grouch have a real name?`
    // `Give a brief precis of the novel: list genre, describe the major characters, and provide an overall plot summary.`
    // `Analyze the story, and let me know if you think this is similar to any other well-known stories in its genre, or in another genre.`
    // `Proofreading: are there any spelling, grammar, or punctuation errors that can distract readers from the story itself?`
    `Character development: Analyze the important characters and assess how well-developed they are, with distinct personalities, backgrounds, and motivations. This will make them more relatable and engaging to readers.`
    // `Find me an extract from the novel that shows character development of the main protagonist.`
    // `Plot structure: Analyze whether the story's events are in a clear and coherent sequence, with rising action, climax, falling action, and resolution. This will help maintain reader interest throughout the novel.`
    // `Subplots: Analyze the sub-plots and minor characters to verify that they add to the story instead of distracting from it. Sub-plots and side-characters should enhance the story and not confuse the reader. Point out any flaws.`
    // `Show, don't tell: Analyze whether the story simply tells readers what is happening or how characters feel, or whether it uses vivid descriptions and actions to show them. This will make the writing more engaging and immersive.`
    // `Consistent point of view: Does the novel stick to one consistent point of view throughout, whether it be first person, third person limited, or omniscient? This will help maintain a cohesive narrative voice.`
    // `Active voice: Does the writing use active voice instead of passive voice whenever possible? This makes the writing more direct and engaging.`
    // `Vary sentence structure: Does the writing break up long sentences with shorter ones to create rhythm and variety? This will make the writing more dynamic and interesting to read.`
    // `Analyze the story from the point of view of a potential reader who purchases the book.`
    // `Analyze the story from the point of view of a literary agent reading this book for the first time and trying to decide if they want to represent this author to publishers.`
    // `Provide suggestions on how to improve any confusing parts of the plot. If there are other narrative elements which should be revised and improved, point them out.`
    // `List all the chapters in the book, and give a one-sentence summary of each chapter.`
    // `What do you dislike the most about the book? What needs fixing most urgently?`
    // `Who is the ideal audience for this book? What will they enjoy about it? What might they dislike about it? How can the story be adjusted to make it appear to a wider audience?`
    // `Identify any repetitive or superfluous elements in the book.`
    // `Identify any subplots which don't lead anywhere and just distract from the main story.`
    // `Write a dust-jacket blurb describing this book, with some invented (but realistic) quotes from reviewers about how good the book is. Do not include any extracts from the book itself.`
    // `Who turns out to have killed Tyler in the end?`
    // `Write query letter to a potential publisher, explaining the novel and its potential market.`
    // `Provide an extract from the novel showing how the murder of Tyler is solved.`
;


console.log(chalk.greenBright(`Question: ${input}`));

// const result = { output: JSON.stringify(await qaChain.invoke(input)) };

// const result = { output: JSON.stringify(await qaRetriever.invoke(input))};

const result = await executor.invoke({ input },
{
    callbacks: [{
        handleToolStart(tool, input, runId, parentRunId, tags, metadata, runName) {
            internLookup[runId] = runName || tool.name || (tool.id && tool.id[2]);
            console.log(chalk.blueBright(`Boss asks ${internLookup[runId]}: ${input}`));
        },
        handleToolEnd(output, runId, parentRunId, tags) {
            console.log(chalk.bgBlueBright(`${internLookup[runId] || 'Intern'} responds: ${output}`));
            delete internLookup[runId];
        },
        handleToolError(err, runId, parentRunId, tags) {
            console.log(chalk.red(`${internLookup[runId] || 'Intern'} errors: ${err.message} : ${JSON.stringify(err)}`), err);
            delete internLookup[runId];
        },
        handleAgentAction(action, runId, parentRunId, tags) {
            const thought = action.log.trim().match(/.*^Thought:(.*?)(?:^Action:|^Final Answer:)/ms)[1].trim();
            if(thought) {
                console.log(chalk.whiteBright(`\n${thought}\n`));
            } else {
                console.log(chalk.whiteBright(`\n${action.log.trim()}`));
            }
        },
        // handleLLMStart(llm, message, runId, parentRunId, extraParams, tags, metadata, runName) {
        //     console.log(chalk.yellowBright(`\nAsking LLM: ${message}`));
        // },
        // handleLLMEnd(llmOutput, runId, parentRunId, tags) {
        //     console.log(chalk.bgYellowBright(`\nLLM result: ${llmOutput.generations[0][0].text}\n`));
        // },
    }],
});

console.log(chalk.greenBright(result.output));
