import chalk from 'chalk';
import _ from 'lodash';

import { OllamaEmbeddings } from '@langchain/community/embeddings/ollama';
import { ChatOllama } from '@langchain/community/chat_models/ollama';
import { Bedrock } from "@langchain/community/llms/bedrock";
import { FaissStore } from '@langchain/community/vectorstores/faiss';
import { HydeRetriever } from "langchain/retrievers/hyde";
import { RunnableSequence, RunnablePassthrough } from '@langchain/core/runnables';
import { DynamicTool } from '@langchain/core/tools';
import { WikipediaQueryRun } from '@langchain/community/tools/wikipedia_query_run';
import { DuckDuckGoSearch } from '@langchain/community/tools/duckduckgo_search';

import { AgentExecutor } from 'langchain/agents';

import { ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate } from '@langchain/core/prompts';

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

// Mistral 7b-instruct in the cloud on amazon - cheap and fast, but not so smrt; 32k context 8k outputs
// const llm = new Bedrock({
//     model: "mistral.mistral-7b-instruct-v0:2", // You can also do e.g. "anthropic.claude-v2"
//     region: "us-west-2",
//     temperature: 0,
//     maxTokens: 8192,
// });

// mistral7b-instruct has 32k training ctx but ollama sets it to 2k so need to override that here
// Prompt parse: ~550-600 t/s; generation: ~50-60 t/s
const fastestLLMChat = new ChatOllama({ model: 'mistral:instruct', temperature: 0, numCtx: 32768, baseUrl: 'http://127.0.0.1:11435' });
const fastestLLMJSON = new ChatOllama({ model: 'mistral:instruct', temperature: 0, numCtx: 32768, format: 'json', baseUrl: 'http://127.0.0.1:11435' });
const fastestLLMInstruct = new ChatOllama({ model: 'mistral:instruct', temperature: 0, numCtx: 32768, baseUrl: 'http://127.0.0.1:11435' });

// llama3 llama3:8b-instruct-q5_K_M has 8k ctx
const llama3LLMChat = new ChatOllama({ model: 'llama3:8b-instruct-q5_K_M', temperature: 0, numCtx: 8192, baseUrl: 'http://127.0.0.1:11435' });
const llama3LLMJSON = new ChatOllama({ model: 'llama3:8b-instruct-q5_K_M', temperature: 0, numCtx: 8192, format: 'json', baseUrl: 'http://127.0.0.1:11435' });
const llama3LLMInstruct = new ChatOllama({ model: 'llama3:8b-instruct-q5_K_M', temperature: 0, numCtx: 8192, baseUrl: 'http://127.0.0.1:11435' });

// mixtral:8x7b-instruct-v0.1-q5_K_M - 32k context
// Prompt parse: ~150-200 t/s; generation: ~20-25 t/s
const slowLLMChat = new ChatOllama({ model: 'mixtral:8x7b-instruct-v0.1-q5_K_M', temperature: 0, numCtx: 32768, baseUrl: 'http://127.0.0.1:11436' });
const slowLLMJSON = new ChatOllama({ model: 'mixtral:8x7b-instruct-v0.1-q5_K_M', temperature: 0, numCtx: 32768, format: 'json', baseUrl: 'http://127.0.0.1:11436' });
const slowLLMInstruct = new ChatOllama({ model: 'mixtral:8x7b-instruct-v0.1-q5_K_M', temperature: 0, numCtx: 32768, baseUrl: 'http://127.0.0.1:11436' });

// command-r:35b-v0.1-q6_K has 128k training ctx but ollama sets it to 2k so need to override that here
// Prompt parse: ~20t/s; generation: ~4 t/s
const commandRLLMChat = new ChatOllama({ model: 'command-r:35b-v0.1-q6_K', temperature: 0, numCtx: 32768, baseUrl: 'http://127.0.0.1:11436' });
const commandRLLMJSON = new ChatOllama({ model: 'command-r:35b-v0.1-q6_K', temperature: 0, numCtx: 32768, format: 'json', baseUrl: 'http://127.0.0.1:11436' });
const commandRLLMInstruct = new ChatOllama({ model: 'command-r:35b-v0.1-q6_K', temperature: 0, numCtx: 32768, baseUrl: 'http://127.0.0.1:11436' });

// mixtral:8x22b-instruct-v0.1-q4_0 - 65k training context but ollama sets it to 2k, has special tokens for tools and shit
// Prompt parse: ~60-80 t/s; generation ~11-12 t/s
const slowestLLMChat = new ChatOllama({ model: 'mixtral:8x22b-instruct-v0.1-q4_0', temperature: 0, numCtx: 65536, baseUrl: 'http://127.0.0.1:11437' });
const slowestLLMJSON = new ChatOllama({ model: 'mixtral:8x22b-instruct-v0.1-q4_0', temperature: 0, numCtx: 65536, format: 'json', baseUrl: 'http://127.0.0.1:11437' });
const slowestLLMInstruct = new ChatOllama({ model: 'mixtral:8x22b-instruct-v0.1-q4_0', temperature: 0, numCtx: 65536, baseUrl: 'http://127.0.0.1:11437' });

// Same guy, but in the cloud - faster but more $
// const llm = new Bedrock({
//     model: "mistral.mixtral-8x7b-instruct-v0:1", // You can also do e.g. "anthropic.claude-v2"
//     region: "us-west-2",
//     temperature: 0,
//     maxTokens: 4096,
// });

const storeDirectory = 'novels/Christmas Town draft 2';
// const storeDirectory = 'novels/Fighters_pages';
const vectorStore = await FaissStore.load(
    storeDirectory,
    embeddings
);

const hydePrompt = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate('Provide 3 alternate phrasings of the query, and then write a short paragraph which responds to the query.'),
    HumanMessagePromptTemplate.fromTemplate('Query: {question}'),
]);

// const retriever = vectorStore.asRetriever({ k: 15 });
const qaRetriever = new HydeRetriever({
    // verbose: true,
    vectorStore,
    llm: llama3LLMChat, // Basic task to write the prompt so do it quickly
    k: 20,
    promptTemplate: hydePrompt,
});

const extractRetriever = new HydeRetriever({
    // verbose: true,
    vectorStore,
    llm: llama3LLMChat, // Basic task to write the prompt so do it quickly
    k: 20,
    promptTemplate: hydePrompt,
});

const mainAgentPromptTemplate = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(
`You are a professional developmental editor, working for the author of a novel to improve the latest draft of their unpublished novel before they submit it to literary agents.

Find the book's flaws when they exist, and help fix them - that is the whole point. Analyze any flaws rigorously and point them out to the author.

In each iteration, you first will come up with a plan or thought, and you will output it like this:

\`\`\`
Thought: you should always think about what additional research might help better answer the question
\`\`\`

After providing the thought, you will choose either to use a tool to gather more information, or whether you have enough information for a final answer.

You have access to the following tools to help with researching your answer:

\`\`\`
{tools}
\`\`\`

Tools do not remember what you've asked them to do before, nor about do they know about your thoughts, they only know about the action input that you send to them for that specific request.
Tools also are terrible at answering multi-part questions. So ask one simple question, get the response, then ask another question, instead of combining questions together in one query.
You should generally not use tools to do creative work, suggest solutions, or do complex analysis. Do that work yourself. Only use tools to gather information about the novel, its characters, plot, scenes and so forth, or to looks things up on the internet.
Do not ask compound questions. Ask one simple thing at a time, but ask as many separate questions as you like successively in subsequent tools calls.

You use a tool by replying like this:

\`\`\`
Action: only the name of the tool to use (should be one of [{tool_names}] verbatim with no escaping characters, omit this "Action" line completely if not requesting an Action)
Action Input: the input to the tool (omit this "Action Input" line completely if not requesting an Action, but ALWAYS include it if you do request an Action)
\`\`\`

The tool will the produce an Observation in response, like this:

\`\`\`
Observation: the result from the tool
\`\`\`

... (you can then repeat this Thought/Action/Action Input/Observation until you have enough information)

Cycle through Though/Action/Observation as many times as necessary - do not take initial tool answers as being exhaustive or conclusive; tools often miss things the first time you ask.

When you are all done and have pursued every thought, and you are not requesting another tools use you may move on to a conclusion as a final answer, but do not do this prematurely - make sure you really thought everything through.

When you have a final answer, you should use this format:

\`\`\`
Final Answer: the final answer to the original query
\`\`\`

Always in your responses then, you should include a single Thought, and either an Action *or* a Final Answer but not both - it's one or the other, but ALWAYS include one of the two.

Here's an example of a full process, using tools, to answer a query:

\`\`\`
Query: Who are the main characters? What language do they speak?
Thought: I should first use a tool to identify the characters, and then use another tool to find something each says and identify the languages.
Action: character-finder-tool
Action Input: Who are the main characters?
Observation: Alison and Mary are the main characters
Thought: I should now use a tool to find something Alison says
Action: quote-finder-tool
Action Input: Please find something Alison says in the book.
Observation: Alison says "I am a girl"
Thought: I can tell that Alison speaks English. Now I should find out what language Mary speaks.
Action: quote-finder-tool
Action Input: Please find something Mary says in the book.
Observation: In chapter 9, Mary says "Je suis une fille"
Thought: Mary speaks French. I now know all the main characters, and their languages, which was the original query.
Final Answer: The main characters are Alison, who speaks English, and Mary, who speaks French.
\`\`\`

`),
        HumanMessagePromptTemplate.fromTemplate(
`Ok, begin!

Query: {input}
{agent_scratchpad}`),
]);

const qaStuffPromptTemplate = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(`You answer queries about a novel in syntactically correct JSON.
You are given extracts from a semantic index of the novel, in JSON.
Consider only these extracts to answer the query. You should ANSWER THE QUERY, and not just regurgitate verbatim quotes.

Make it clear that you're only reading a few shorts extracts from the novel, and you might be overlooking parts of the novel that might shed more light on the query.
Let your boss know that they can ask you to read further if necessary to confirm things.

If you don't know the answer, or the answer cannot be found in the extracts, just say that you don't know for sure, and include your best guess.
You won't get in trouble for saying you don't know. You can do that by responding like this for example:

\`\`\`
{{"warning":"I can only make an educated guess based on the few short passages that I read.","answer":"As far as I can tell, the hero dies at the end."}}
\`\`\`

If the question is too vague or complex, then ask for it to be broken down into simpler questions or to be rephrased in a more precise way.
You can do that like this for example:

\`\`\`
{{"warning":"That question is too complex for me. Could you break it down into simpler questions, and ask them one at a time?"}}
\`\`\`

When you respond, you *must* do so in syntactically correct JSON for example:

\`\`\`
{{"answer":"The hero starts off at home (page 17), later in the novel he is in a car (page 211). Later still, he gets to the office (page 300)."}}
\`\`\``),
    HumanMessagePromptTemplate.fromTemplate(`Query: {question}
Extracts:
{context}`),
]);

const extractStuffPromptTemplate = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(`You provide verbatim extracts or passages (up to a paragraph or so long) from a novel.

You are given extracts from a semantic index of the novel in JSON format.

Given these extracts and a query, return a single extract (up to a paragraph or so long) from the novel that is the most relevant to the query.

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
{{"warning":"No single extract really covers everything you asked about. Please ask again focusing on any areas this extract doesn't cover","commentary":"This extract is a perfect example of symbolism used in the text, which is part of what you were asking for.","extract":{{"loc":{{"lines":{{"from":423,"to":551}}}},"text":"Inside, the glass walls reflected not just the world outside but also the soul within, as if inviting visitors to see beyond the veil of reality and touch the very essence of their own being."}}}}
\`\`\`

When you find an ideal extract, omit any warning, but you should include your own commentary on the extract, along with the verbatim text.
The commentary should clarify the context of the extract, and identify any pronouns used.
All responses *must* be in syntactically valid JSON for example:

\`\`\`
{{"commentary":"I chose this extract because it shows clearly how the hero defeated the dragon. \\"He\\" in the extract refers to the hero.","extract":{{"loc":{{"lines":{{"from":123,"to":456}}}},"text":"And then, he stabbed the dragon straight through the heart and killed it."}}}}
\`\`\`

When there is an appropriate extract, always return it as demonstrated above. Do not forget to include the actual extract from your response when there is one, and always use the most representative extract that best responds to the query.`,
    ),
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
        name: 'Joe Analyst',
        description: 'An intern who answers questions about what happens in the novel. He is best for broad questions about the novel.',
        func: async (x) => JSON.stringify(await qaChain.invoke(x)),
    }),
    new DynamicTool({
        // verbose: true,
        name: 'Sally Extractor',
        description: 'An intern who can read the novel and provide a single short extract up to a paragraph or so long. She is best for very specific verbatim extracts about specific details.',
        func: async (x) => JSON.stringify(await extractRetrievalChain.invoke(x)),
    }),
    // new WikipediaQueryRun({
    //     topKResults: 3,
    //     maxDocContentLength: 4000,
    // }),
    new DuckDuckGoSearch({
        maxResults: 3
    }),
    // new ExaSearchResults({
    //     client: new Exa('7980b8df-2d60-4900-bc29-4d3695eb4e45'),
    // }),
];

const agent = await createReactAgent({
    llm: slowLLM,
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
            console.log(chalk.red(`${internLookup[runId] || 'Intern'} errors: ${err.message} : ${JSON.stringify(err)}`));
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
