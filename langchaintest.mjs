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
import { WebBrowser } from 'langchain/tools/webbrowser';

import { AgentExecutor } from 'langchain/agents';

import { ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate } from '@langchain/core/prompts';

import { renderTextDescription } from 'langchain/tools/render';
import { ReActSingleInputOutputParser } from './node_modules/langchain/dist/agents/react/output_parser.js';
import { AgentRunnableSequence } from './node_modules/langchain/dist/agents/agent.js';
import { StringOutputParser, JsonOutputParser } from '@langchain/core/output_parsers';
import { formatLogToString } from './node_modules/langchain/agents/format_scratchpad/log.js';

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
    // const llmWithStop = llm.bind({
    //     stop: ["\nObservation:"],
    // });
    const agent = AgentRunnableSequence.fromRunnables([
        RunnablePassthrough.assign({
            agent_scratchpad: (input) => formatLogToString(input.steps, undefined, 'Now continue!\n'),
        }),
        partialedPrompt,
        llm,
        new ReActSingleInputOutputParser({
            toolNames,
        }),
    ], {
        name: "ReactAgent",
        streamRunnable,
        singleAction: true,
    });
    return agent;
}

// const embeddings = new OllamaEmbeddings({ model: 'nomic-embed-text', numCtx: 2048 });
const embeddings = new OllamaEmbeddings({ model: 'mxbai-embed-large', numCtx: 512 });
// const embeddings = new OllamaEmbeddings({ model: 'mistral:7b-instruct-v0.2-fp16', numCtx: 32768 });

const commonOptions = { temperature: 0, seed: 19740822 };
const commonOptionsJSON = { temperature: 0, format: 'json' };
const commonOptions8k = { numCtx: 8 * 1024, ...commonOptions };
const commonOptions8kJSON = { ...commonOptions8k, ...commonOptionsJSON };
const commonOptions32k = { numCtx: 32 * 1024, ...commonOptions };
const commonOptions32kJSON = { ...commonOptions32k, ...commonOptionsJSON };
const commonOptions64k = { numCtx: 64 * 1024, ...commonOptions };
const commonOptions64kJSON = { ...commonOptions64k, ...commonOptionsJSON };

// mistral:7b-instruct-v0.2 has 32k training ctx but ollama sets it to 2k so need to override that here
// Prompt parse: ~550-600 t/s; generation: ~50-60 t/s
const mistralLLMChat = new ChatOllama({ model: 'mistral:7b-instruct-v0.2-q8_0', ...commonOptions32k });
const mistralLLMJSON = new ChatOllama({ model: 'mistral:7b-instruct-v0.2-q8_0', ...commonOptions32kJSON });

// llama3.1 has 128k ctx
// Prompt parse: ~500 t/s; generation: ~40 t/s
const llama3_8bLLMChat = new ChatOllama({ model: 'llama3.1:8b-instruct-q8_0', ...commonOptions64k });
const llama3_8bLLMJSON = new ChatOllama({ model: 'llama3.1:8b-instruct-q8_0', ...commonOptions64kJSON });

// phi3:medium-128k-instruct-q8_0 has 128k ctx but we'll only use 64k
// Prompt parse: ~250 t/s; generation: ~20 t/s
const phi3_14bLLMChat = new ChatOllama({ model: 'phi3:medium-128k-instruct-q8_0', ...commonOptions64k });
const phi3_14bLLMJSON = new ChatOllama({ model: 'phi3:medium-128k-instruct-q8_0', ...commonOptions64kJSON });

// mixtral:8x7b-instruct-v0.1-q8_0 - 32k context
// Prompt parse: ~200 t/s; generation: ~20-25 t/s
const mixtral7BLLMChat = new ChatOllama({ model: 'mixtral:8x7b-instruct-v0.1-q8_0', ...commonOptions32k });
const mixtral7BLLMJSON = new ChatOllama({ model: 'mixtral:8x7b-instruct-v0.1-q8_0', ...commonOptions32kJSON });

// llama3.1 has 128k ctx
// Prompt parse: ~65 t/s; generation: ~4-6 t/s
const llama3_70bLLMChat = new ChatOllama({ model: 'llama3.1:70b-instruct-q8_0', ...commonOptions64k });
const llama3_70bLLMJSON = new ChatOllama({ model: 'llama3.1:70b-instruct-q8_0', ...commonOptions64kJSON });

// llama3-chatqa llama3-chatqa:70b-v1.5-q8_0 has 32k ctx
// It wants special format with system (context appended to system), then user
// Prompt parse: ~65 t/s; generation: ~4-6 t/s
// const llama3chatQA_70bLLMChat = new ChatOllama({ model: 'llama3-chatqa:70b-v1.5-q8_0', ...commonOptions32k });
// const llama3chatQA_70bLLMJSON = new ChatOllama({ model: 'llama3-chatqa:70b-v1.5-q8_0', ...commonOptions32kJSON });

// mixtral:8x22b-instruct-v0.1-q4_0 - 64k training context but ollama sets it to 2k, has special tokens for tools and shit
// Prompt parse: ~60-80 t/s; generation ~11-12 t/s
const mixtral22bLLMChat = new ChatOllama({ model: 'mixtral:8x22b-instruct-v0.1-q4_0', ...commonOptions64k });
const mixtral22bLLMJSON = new ChatOllama({ model: 'mixtral:8x22b-instruct-v0.1-q4_0', ...commonOptions64kJSON });

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
Provide 3 alternate phrasings of the query (with no formatting, bullets or numbering), and then write a short paragraph which responds to the query.

## Example:

User: What's your name?

Assistant: What do you call yourself?
What is your name?
How do you say your name?

My name is Simon Smith.`),
    HumanMessagePromptTemplate.fromTemplate('{question}'),
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
    llm: llama3_8bLLMChat, // Basic task to write the prompt so do it quickly
    searchType: 'mmr',
    searchKwargs: {
        lambda: 0.5,
        fetchK: 200,
    },
    k: 100,
    promptTemplate: hydePrompt,
});
qaRetriever._getRelevantDocuments = mmrSearch.bind(qaRetriever);

const extractRetriever = new HydeRetriever({
    // verbose: true,
    vectorStore,
    llm: llama3_8bLLMChat, // Basic task to write the prompt so do it quickly
    searchType: 'mmr',
    searchKwargs: {
        lambda: 0.5,
        fetchK: 200,
    },
    k: 100,
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
Here is a complete list of tools that you have available to you:

{tools}

There are no other tools. Anything else that needs doing, you'll have to do yourself.

# How to proceed with your work

1. You will use a sequence of individual steps to answer the query.

Every response will first *always* begin with a thought for how to achieve this step, and you will output it like this:

\`\`\`
Thought: you should always think about what additional research might help better answer the question
\`\`\`

2. After providing the thought, you will decide either to use a tool to gather more information, or that you have enough information already for a final answer.

## If you need to use a tool
Tools are stateless, and they do not themselves have access to tools, they only know about the action input that you send to them for that specific request.
Tools also are terrible at answering multi-part questions. So ask one simple question, get the response, then ask another question, instead of combining questions together in one query.
You should generally not use tools to do creative work, suggest solutions, or do complex analysis. Do that work yourself.
Do not ask compound questions. Ask one simple thing at a time, but ask as many separate questions as you like successively in subsequent tools calls.

You use a tool by including this in your reply after the Thought with a tool name selected from {tool_names}:

\`\`\`
Action: {{
  "tool": <name of the selected tool>,
  "tool_input": <parameters for the selected tool, matching the tool's JSON schema>
}}
\`\`\`

3. The tool will the produce an Observation in response, like this:

\`\`\`
Observation: the result from the tool
\`\`\`

Observations are generated only by tools, not by the Assistant. You should never generate your own Observations, let the tools do that for you.

If the observation reports that something wasn't clear, or couldn't be determined from the input, you can call the tool again to follow-up and ask for more information.

4. ... (you can then repeat this Thought/Action/Observation until you have enough information). Cycle through Though/Action/Observation as many times as necessary - do not take initial tool answers as being exhaustive or conclusive; tools often miss things the first time you ask.

5. When you are all done and have pursued every thought, and you are not requesting another tools use you may move on to a conclusion as a final answer, but do not do this prematurely - make sure you really thought everything through.
When you have a final answer, you should append a Final Answer after your Thought:

\`\`\`
Final Answer: the final answer to the original query
\`\`\`

## Remember - critical!
Response the pattern should be: Thought (required every time), Action (optional) many times with the tool providing Observations, then Final Answer (at the very end)
Always in your responses you should include:
  - a single Thought
  - either an Action *or* a Final Answer but not both - it's one or the other, but ALWAYS include one of the two.

## Examples:

User:
\`\`\`
Query: Some question about the novel
\`\`\`

Assistant:
\`\`\`
Thought: An overall plan, broken down into steps. You will then iterate through the steps one by one to compile a final answer.
Action: {{
  "tool": <name of the selected tool>,
  "tool_input": <parameters for the selected tool, matching the tool's JSON schema>
}}
\`\`\`

Tool output:
\`\`\`
Observation: results from the tool
\`\`\`

... (more thought/action/observations)

Eventually:
\`\`\`
Thought: Ok, from all the observations so far, I'm ready to produce a final result
Final Answer: The best answer to the query based on the thoughts and observations above.
\`\`\`

{agent_scratchpad}`),
    HumanMessagePromptTemplate.fromTemplate(`{input}`),
    HumanMessagePromptTemplate.fromTemplate(`What's your next Thought and *either* Action, *or* Final Answer (include one of Action or Final Answer each time)?`),
]);

const qaStuffPromptTemplate = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(`# Basic Rules
You work as an assistant to a developmental editor (as user), who in turn is assisting authors to improve their unpublished novels. When you answer the user's requests, you cite your sources in your answers.

# Task and Context
You help users by answering their queries. You will be asked a very wide array of requests on all kinds of topics. You should focus on serving the user's needs as best you can, which will be wide-ranging.
You are given a limited number of chunks retrieved from a semantic-indexed database of a novel. Answer the query as completely and accurately as possible by considering these extracts. You should ANSWER THE QUERY, and not just regurgitate verbatim quotes.

# Unable to answer
If you are unable to answer for some reason, or only able to provide a partial answer, then explain this to the user by including a \`warning\` property in your response, indicating the problem, and what the user can to do help resolve the issue.

# Reply in JSON
It is very important that your replies be in the form of a JSON object conforming to this JSON schema:

{{
  "$id": "https://example.com/person.schema.json",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Response",
  "type": "object",
  "properties": {{
    "answer": {{
      "type": "string",
      "description": "The answer to the user's query"
    }},
    "warning": {{
      "type": "string",
      "description": "Any warnings about the answer, or any other information to help the user resolve the issue"
    }},
    "required": []
  }}
}}

{context}`),
    HumanMessagePromptTemplate.fromTemplate(`{question}`),
]);

const extractStuffPromptTemplate = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(`# Basic rules
You provide verbatim extracts or passages from a novel for a developmental editor.
Given a set of extracts and a query, return a single extract from the novel that is the most relevant to the query.

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
All responses *must* be in syntactically valid JSON conforming to the following JSON schema:

{{
  "$id": "https://example.com/person.schema.json",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Response",
  "type": "object",
  "properties": {{
    "extract": {{
      "type": "object",
      "description": "The single extract most relevant to the user's query"
      "properties": {{
        "text": {{
            "type": "string",
            "description": "The verbatim extract from the novel"
        }},
        "loc": {{
            "type": "object",
            "properties": {{
                "lines": {{
                    "type": "object",
                    "properties": {{
                        "from": {{
                            "type": "integer"
                        }},
                        "to": {{
                            "type": "integer"
                        }}
                    }}
                }},
                "page": {{
                    "type": "integer"
                }}
            }}
        }}
      }}
    }},
    "commentary": {{
      "type": "string",
      "description": "Your commentary on why the chosen extract was selected"
    }},
    "warning": {{
      "type": "string",
      "description": "Any warnings about the answer, or any other information to help the user resolve the issue"
    }},
    "required": []
  }}
}}

{context}`),
    HumanMessagePromptTemplate.fromTemplate(`{question}`),
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

    llama3_8bLLMJSON,

    new JsonOutputParser(),
]);

const extractRetrievalChain = RunnableSequence.from([
    {
        context: extractRetriever.pipe(sortDocsFormatAsJSON),
        question: new RunnablePassthrough(),
    },

    extractStuffPromptTemplate,

    llama3_8bLLMJSON,

    new JsonOutputParser(),
]);

const tools = [
    new DynamicTool({
        // verbose: true,
        name: 'novel-analyst',
        description: 'A tool that can answer basic questions about story elements of the novel: plot, characters, scenes and such. Tool input should be questions written in complete sentences.',
        func: async (x) => JSON.stringify(await qaChain.invoke(JSON.stringify(x), {
            callbacks: [{
                handleLLMStart(llm, message, runId, parentRunId, extraParams, tags, metadata, runName) {
                    console.log(chalk.yellowBright(`\nAsking LLM: ${message}`));
                },
                handleLLMEnd(llmOutput, runId, parentRunId, tags) {
                    console.log(chalk.bgYellowBright(`\nLLM result: ${llmOutput.generations[0][0].text}\n`));
                },
            }],
        })),
    }),
    new DynamicTool({
        // verbose: true,
        name: 'quote-extractor',
        description: 'A tool that provides a single short extract up to at most a paragraph or so long. It is best for very specific verbatim extracts about specific things that happen in the novel. Tool input should be in complete sentences.',
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
    new DuckDuckGoSearch({
        maxResults: 3
    }),
    new WebBrowser({
        model: llama3_8bLLMChat,
        embeddings,
    }),
    // new ExaSearchResults({
    //     client: new Exa('7980b8df-2d60-4900-bc29-4d3695eb4e45'),
    // }),
];

const agent = await createReactAgent({
    llm: llama3_8bLLMChat,
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

const input =
    // `What do you think of this novel?`
    // `Identify sentences which are too long or complex and are hard to understand.`
    // `What would be a good, engaging title for this novel?`
    // `Give a precis of the novel: list genre, describe the protagonist and major characters, and provide an overall plot summary.`
    // `Analyze the story, and let me know if you think this is similar to any other well-known stories in its genre, or in another genre.`
    // `Proofreading: are there any spelling, grammar, or punctuation errors that can distract readers from the story itself?`
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
    // `Write a dust-jacket blurb describing this novel and a plot synopsis, in an engaging way but without spoilers, with some invented (but realistic) quotes from reviewers about how good the book is. Do not include any extracts from the book itself. Use markdown syntax for formatting.`
    // `Write a detailed query letter to a potential literary agent, explaining the novel and how it would appeal to readers. The letter should be engaging, and should make the agent interested in repping the book, without being overly cloying or sounding desperate. Be sure to properly research the book content so you're not being misleading. Find out the names of any characters mentioned. The agent will not have read the novel, so any discussion of the novel should not assume that the agent has read it yet. Reference the major events that happen in the book, describe what makes the protagonist engaging for readers, and include something about why the author chose to write this story.`
    // `Is Meghan a likable and relatable character for readers, even if characters in the book perhaps dislike her? Will readers be able to empathize with her and enjoy the novel with her as the protagonist?`
    // `Does Bathrobe Grouch have a real name?`
    // `What is Mr Zimmerman's nickname?`
    `How does it turn out that Tyler died in the end? Who is responsible?`
    // `What is the age of the main character and what are some of the challenges she faces throughout the novel?`
    // `How can the story be adjusted to make it appealing to a wider audience without losing its core themes of trauma, loss, and redemption?`
    // `Are there any secondary characters or subplots in the novel that could be expanded upon to provide additional perspectives or interests?`
    // `Can you provide more context about Roger and his role in the novel? How does he relate to the themes of family, loss, and personal growth?`
    // `Can you provide a brief overview of the main plot points?`
    // `Pick any quotation from the book and count the number of words in it.`
;


console.log(chalk.greenBright(`Question: ${input}`));

// const result = { output: JSON.stringify(await qaChain.invoke(input, {
//     callbacks: [{
//         handleLLMStart(llm, message, runId, parentRunId, extraParams, tags, metadata, runName) {
//             console.log(chalk.yellowBright(`\nAsking LLM: ${message}`));
//         },
//         handleLLMEnd(llmOutput, runId, parentRunId, tags) {
//             console.log(chalk.bgYellowBright(`\nLLM result: ${llmOutput.generations[0][0].text}\n`));
//         },
//     }],
// })) };

// const result = {
//     output: JSON.stringify(await extractRetrievalChain.invoke(input, {
//         callbacks: [{
//             handleLLMStart(llm, message, runId, parentRunId, extraParams, tags, metadata, runName) {
//                 console.log(chalk.yellowBright(`\nAsking LLM: ${message}`));
//             },
//             handleLLMEnd(llmOutput, runId, parentRunId, tags) {
//                 console.log(chalk.bgYellowBright(`\nLLM result: ${llmOutput.generations[0][0].text}\n`));
//             },
//         }],
//     }))
// };

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
            const thoughtParse = action.log.trim().match(/.*^Thought:(.*?)(?:^Action:|^Final Answer:)/ms);
            const thought = thoughtParse && thoughtParse[1] && thoughtParse[1].trim && thoughtParse[1].trim();
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
