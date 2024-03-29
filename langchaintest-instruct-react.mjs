import chalk from 'chalk';

import { OllamaEmbeddings } from '@langchain/community/embeddings/ollama';
import { Ollama } from '@langchain/community/llms/ollama';
import { Bedrock } from "@langchain/community/llms/bedrock";
import { FaissStore } from '@langchain/community/vectorstores/faiss';

import { RetrievalQAChain, MapReduceDocumentsChain, StuffDocumentsChain, LLMChain, loadQARefineChain, loadQAStuffChain } from 'langchain/chains';
import { DynamicTool } from '@langchain/core/tools';

import { AgentExecutor } from 'langchain/agents';

import { PromptTemplate } from '@langchain/core/prompts';

import { RunnablePassthrough, RunnableSequence, } from "@langchain/core/runnables";
import { renderTextDescription } from "langchain/tools/render";
import { ReActSingleInputOutputParser } from "./node_modules/langchain/dist/agents/react/output_parser.js";
import { RunnableSingleActionAgent } from "./node_modules/langchain/dist/agents/agent.js";

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

const embeddings = new OllamaEmbeddings({ model: 'nomic-embed-text', numCtx: 2048 });

// Mistral 7b-instruct in the cloud on amazon - cheap and fast, but not so smrt; 32k context 8k outputs
// const llm = new Bedrock({
//     model: "mistral.mistral-7b-instruct-v0:2", // You can also do e.g. "anthropic.claude-v2"
//     region: "us-west-2",
//     temperature: 0,
//     maxTokens: 8192,
// });

// mistral-instruct-v0.2-2x7b-moe has 32k context, instruct model
const llm = new Ollama({ model: 'cas/mistral-instruct-v0.2-2x7b-moe', temperature: 0, numCtx: 32768 });

// mix-crit is from mixtral:8x7b-instruct-v0.1-q3_K_L - 32k context, instruct model
// const llm = new Ollama({ model: 'mix-crit', temperature: 0, numCtx: 32768 });

// Same guy, but in the cloud - faster but more $
// const llm = new Bedrock({
//     model: "mistral.mixtral-8x7b-instruct-v0:1", // You can also do e.g. "anthropic.claude-v2"
//     region: "us-west-2",
//     temperature: 0,
//     maxTokens: 4096,
// });

const storeDirectory = 'novels/Exile draft 2';
// const storeDirectory = 'novels/Fighters_pages';
const vectorStore = await FaissStore.load(
    storeDirectory,
    embeddings
);

const [B_INST, E_INST] = ['<s>[INST]', '[/INST]'];
const PROMPT = `You are a professional developmental editor, working for the author of a novel to help them improve drafts of their unpublished novel before they submit it to literary agents hoping to get published.

The author will not see anything except for your final answer. Everything else is just internal work which they won't see.

You can assume that the author is intimately familiar with the entire novel that they wrote. Your access to the novel is only through your interns though - you are too important to read it directly yourself.

You do not need to be overly fawning or laud the author if their work is not good; find the book's flaws when they exist, and help fix them.

Comprehensively answer the following question from the author as best you can with reference to the text where appropriate; when referencing the text, include extracts in your answer. Confirm all facts and assumptions through your interns before stating them. When you quote from the novel text, use triple-backticks around the quote like this:

\`\`\`
<here is the text from the novel>
\`\`\`

You have interns to help with researching your answer; interns are quite dumb and have no memory and do not remember what you've asked them to do before, they only know about the question that you send to them for that specific request - so make the questions long and detailed an include all the context that they need within the question. Interns like long detailed questions, the longer and the more detailed, the better! They will also not read the entire novel, but only some sections of it which seem semantically relevant to your question. There is no real way to reference which sections. If you ask an intern the same question multiple times, you will get exactly the same answer each time. Rephrase the question to get different answers. These are the available interns:

{tools}

Use the following format to make use of interns:

\`\`\`
Thought: you should always think about what to do next in the way of any additional research
Action: only the name of the intern to use (should be one of [{tool_names}] verbatim with no escaping characters)
Action Input: the input to the intern
Observation: the output from the intern (wait for the intern to respond)
\`\`\`

... (this Thought/Action/Action Input/Observation can repeat N times)

You should generally not use interns to do creative work, suggest solutions, or do complex analysis. Do that work yourself without using any tool. Only use interns to gather information about the novel, its characters, plot, scenes and so forth.

When you are all done with your research, and done with having interns do their work, and are not requesting another Action you may move on to a final answer, but do not do this pre-maturely.
When you have a final answer, you should use this format:

\`\`\`
Thought: I now know the final answer
Final Answer: the final answer to the original input question
\`\`\`

Never include both an "Action:" and a "Final Answer:" - it's one or the other, but ALWAYS include one of the two.

Begin!

Question: {input}

Previous analysis:
{agent_scratchpad}
`;

const prompt = PromptTemplate.fromTemplate(`${B_INST}${PROMPT}${E_INST}`);

const qaStuffPromptTemplate = PromptTemplate.fromTemplate(`${B_INST}You are an intern who answers questions for a developmental editor.

You found the following extracts from a semantic index of the novel, read only these extracts to answer the question in your own words. There might be other relevant parts, but you didn't read them.

You should digest these extracts and ANSWER THE QUESTION, and not just regurgitate verbatim quotes.

Make it clear in every answer that you're only reading a few shorts extracts from the novel, and you might be overlooking parts of the novel which you didn't actually read. Let your boss know that they can ask you to read further if necessary to confirm things.

If you don't know the answer, or the answer cannot be found in the extracts, just say that you don't know, don't try to make up an answer. You won't get in trouble for saying you don't know. you can do that by responding like this (and not including an extract), for example:

\`\`\`
I'm sorry, but I don't really know how to answer that question from the short passages that I read. Can you give me more guidance as to what other parts of the novel to read?
\`\`\`

If the question is too vague or complex, then ask for it to be broken down into simpler questions or to be more precise. You are encouraged to learn by asking questions. You can do that like this (and without including an extract), for example:

\`\`\`
I didn't really understand that question, it's too complex. Could you break it down into simpler questions, and ask them one at a time?
\`\`\`

Editor's question: {question}

Extracts:
{context}${E_INST}`);

const mapReduceMapPrompt = PromptTemplate.fromTemplate(`${B_INST}You are an intern who follows directions as precisely as possible, working for a developmental editor.
Making no attempt to actually answer the question, just determine whether the extract is relevant to the question. If it is relevant, then return the extract verbatim.

Question: {question}
Extract:
{context}${E_INST}`);

const mapReduceQAReducePrompt = PromptTemplate.fromTemplate(`${B_INST}You are an intern who answers questions for a developmental editor.

You have read only the following extracts from the novel to answer the question in your own words. There might be other relevant part, but you didn't read them.

You should digest these extracts and ANSWER THE QUESTION, and not just regurgitate verbatim quotes.

Make it clear in every answer that you're only reading a few shorts extracts from the novel, and you might be overlooking parts of the novel which you didn't actually read. Let your boss know that they can ask you to read further if necessary to confirm things.

If you don't know the answer, or the answer cannot be found in the extracts, just say that you don't know, don't try to make up an answer. You won't get in trouble for saying you don't know.

If the question is too vague or complex, then ask for it to be broken down into simpler questions or to be more precise. You are encouraged to learn by asking questions.

Editor's question: {question}

Extracts:
{summaries}${E_INST}`);
const mapReduceExtractReducePrompt = PromptTemplate.fromTemplate(`${B_INST}You are an intern whose job is to provide a single verbatim extract or passage from a novel, working for a developmental editor.

Given the following extracts of the novel and a question, return a condensed extract from the novel that is the most relevant to the question.

Make it clear this is only one single extract, and there might be others that are relevant, but you're lazy so you're only returning one. Let your boss know that they can ask for other extracts because you only provide one at a time.

If there are no relevant extracts, just say so. Don't try to make up an answer. You won't get in trouble for saying you don't know. You can do that by responding like this (omitting any extract), for example:

\`\`\`
My commentary: I could not find any relevant extract, sorry! Perhaps try re-phrasing the question?
\`\`\`

If the question is too vague or complex, then ask for it to be broken down into simpler questions or to be more precise. You are encouraged to learn by asking questions. You can do that by responding like this (omitting any extract), for example:

\`\`\`
My commentary: The question is vague, so I'm not sure how to answer it. Please could you be more precise in what you'd like me to find?
\`\`\`

If you are asked for more than one extract, then let your boss know that you can only provide one extract at a time, and for further extracts, they should rephrase the question (and that you are only including a single extract), for example:

\`\`\`
My commentary: I can only provide a single extract, but you asked for more than one. Please ask for only one at a time, and rephrase the question each time to get more.
\`\`\`

When returning an extract, you can include your own commentary on the extract, along with the verbatim text, for example:

\`\`\`
My commentary: This extract shows how things work
Verbatim extract: The thingamajig slots into the socket, and then turns to power the motor.
\`\`\`

Question: {question}
Extracts:
{summaries}${E_INST}`);

const refineQAInitialPrompt = PromptTemplate.fromTemplate(`${B_INST}You are an intern who follows directions as precisely as possible, working for a developmental editor. Answer the question, using only the context.

Given the following extracted parts of the novel and a question, answer the question.

If you don't know the answer, just say that you don't know. Don't try to make up an answer. You won't get in trouble for saying you don't know.

If the question is too vague or complex, then ask for it to be broken down into simpler questions or to be more precise. You are encouraged to learn by asking questions.

Question: {question}

Context:
{context}${E_INST}`);
const refineQARefinePrompt = PromptTemplate.fromTemplate(`${B_INST}You are an intern who follows directions as precisely as possible, working for a developmental editor.

Your boss the editor has asked the following question: {question}

If the question is too vague or complex, then ask for it to be broken down into simpler questions or to be more precise. You are encouraged to learn by asking questions.

You have provided an existing answer:
------------
{existing_answer}
------------

You have the opportunity to refine the existing answer (only if needed) with some more context below:
------------
{context}
------------

Given the new context, refine the original answer to better answer the question.
If the context isn't useful, return the original answer.${E_INST}`);

const qaChain = new RetrievalQAChain({
    // verbose: true,
    retriever: vectorStore.asRetriever(25),
    // combineDocumentsChain: loadQAMapReduceChain(llm, {
    //     verbose: true,
    //     combineMapPrompt: mapReduceMapPrompt,
    //     combinePrompt: mapReduceQAReducePrompt,
    // }),

    // combineDocumentsChain: loadQARefineChain(llm, {
    //     verbose: true,
    //     questionPrompt: refineQAInitialPrompt,
    //     refinePrompt: refineQARefinePrompt,
    // }),

    combineDocumentsChain: loadQAStuffChain(llm, {
        // verbose: true,
        prompt: qaStuffPromptTemplate,
    }),

    llm,
});

function loadQAMapReduceChain(llm, params = {}) {
    const { combineMapPrompt, combinePrompt, verbose, combineLLM, returnIntermediateSteps, maxTokens, maxIterations, ensureMapStep, } = params;
    const llmChain = new LLMChain({ prompt: combineMapPrompt, llm, verbose });
    const combineLLMChain = new LLMChain({
        prompt: combinePrompt,
        llm: combineLLM ?? llm,
        verbose,
    });
    const combineDocumentChain = new StuffDocumentsChain({
        llmChain: combineLLMChain,
        documentVariableName: "summaries",
        verbose,
    });
    const chain = new MapReduceDocumentsChain({
        llmChain,
        combineDocumentChain,
        returnIntermediateSteps,
        verbose,
        maxTokens,
        maxIterations,
        ensureMapStep,
    });
    return chain;
}

const extractRetrievalChain = new RetrievalQAChain({
    // verbose: true,
    retriever: vectorStore.asRetriever(20),

    combineDocumentsChain: loadQAMapReduceChain(llm, {
        // verbose: true,
        combineMapPrompt: mapReduceMapPrompt,
        combinePrompt: mapReduceExtractReducePrompt,
        ensureMapStep: false,
        maxTokens: 28000, // Leave some headroom for instructions, etc
        maxIterations: 1,
    }),

    // combineDocumentsChain: loadQARefineChain(llm, {
    //     verbose: true,
    //     questionPrompt: refineQAInitialPrompt,
    //     refinePrompt: refineQARefinePrompt,
    // }),

    // combineDocumentsChain: loadQAStuffChain(llm, {
    //     verbose: true,
    //     prompt: qaStuffPromptTemplate,
    // }),

    llm,
});

const tools = [
    new DynamicTool({
        // verbose: true,
        name: 'Joe Analyst',
        description: 'A junior intern who answers questions about what happens after reading only some sections of the novel that he thinks might be relevant, and not the entire novel. He will let you know if he thinks there are problems with his answer. He is best for broad questions about the novel.',
        func: async (x) => (await qaChain.invoke({ query: x })).text,
    }),
    new DynamicTool({
        // verbose: true,
        name: 'Sally Extractor',
        description: 'A junior intern who can read the novel and provide a single short extract that demonstrates a relevant semantic concept. She will let you know if she thinks there are problems with her answer. She is best for very specific extracts about specific details. If you need multiple extracts, you should ask her for only one at a time, and ask her multiple times with rephrased questions each time.',
        func: async (x) => (await extractRetrievalChain.invoke({ query: x })).text,
    }),
    // new WikipediaQueryRun({
    //     topKResults: 3,
    //     maxDocContentLength: 4000,
    // }),
    // new ExaSearchResults({
    //     client: new Exa('7980b8df-2d60-4900-bc29-4d3695eb4e45'),
    // }),
];

// console.log(tools);
// const result = retrievalQAChain.invoke({ query: `Who are the main characters in the novel?` });

const agent = await createReactAgent({
    llm,
    tools,
    prompt,
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
    // `Give a brief precis of the novel: list genre, describe the major characters, and provide an overall plot summary.`
    // `Analyze the story, and let me know if you think this is similar to any other well-known stories in its genre, or in another genre.`
    // `Proofreading: are there any spelling, grammar, or punctuation errors that can distract readers from the story itself?`
    // `Character development: Analyze the important characters and assess how well-developed they are, with distinct personalities, backgrounds, and motivations. This will make them more relatable and engaging to readers.`
    // `Plot structure: Analyze whether the story's events are in a clear and coherent sequence, with rising action, climax, falling action, and resolution. This will help maintain reader interest throughout the novel.`
    // `Subplots: Analyze the sub-plots and minor characters to verify that they add to the story instead of distracting from it. Sub-plots and side-characters should enhance the story and not confuse the reader. Point out any flaws.`
    // `Show, don't tell: Analyze whether the story simply tells readers what is happening or how characters feel, or whether it uses vivid descriptions and actions to show them. This will make the writing more engaging and immersive.`
    // `Consistent point of view: Does the novel stick to one consistent point of view throughout, whether it be first person, third person limited, or omniscient? This will help maintain a cohesive narrative voice.`
    // `Active voice: Does the writing use active voice instead of passive voice whenever possible? This makes the writing more direct and engaging.`
    `Vary sentence structure: Does the writing break up long sentences with shorter ones to create rhythm and variety? This will make the writing more dynamic and interesting to read.`
    // `Analyze the story from the point of view of a potential reader who purchases the book.`
    // `Analyze the story from the point of view of a literary agent reading this book for the first time and trying to decide if they want to represent this author to publishers.`
    // `Provide suggestions on how to improve any confusing parts of the plot. If there are other narrative elements which should be revised and improved, point them out.`
    // `List all the chapters in the book, and give a one-sentence summary of each chapter.`
    // `What do you dislike the most about the book? What needs fixing most urgently?`
    // `Who is the ideal audience for this book? What will they enjoy about it? What might they dislike about it? How can the story be adjusted to make it appear to a wider audience?`
    // `Identify any repetitive or superfluous elements in the book.`
    // `Identify any subplots which don't lead anywhere and just distract from the main story.`
;


console.log(chalk.greenBright(`Question: ${input}`));

const result = await executor.invoke({ input },
{
    callbacks: [{
        handleToolStart(tool, input, runId, parentRunId, tags, metadata, runName) {
            internLookup[runId] = runName;
            console.log(chalk.blueBright(`Boss asks ${runName}: ${input}`));
        },
        handleToolEnd(output, runId, parentRunId, tags) {
            console.log(chalk.blue(`${internLookup[runId] || 'Intern'} responds: ${output}`));
            internLookup[runId] = undefined;
        },
        handleAgentAction(action, runId, parentRunId, tags) {
            const thought = action.log.trim().match(/.*^Thought:(.*?)(?:^Action:|^Final Answer:)/ms)[1].trim();
            console.log(chalk.whiteBright(`\n${thought}\n`));
        },
        // handleLLMStart(llm, message, runId, parentRunId, extraParams, tags, metadata, runName) {
        //     console.log(chalk.yellowBright(`\nAsking LLM: ${message}`));
        // },
        // handleLLMEnd(llmOutput, runId, parentRunId, tags) {
        //     console.log(chalk.yellow(`\nLLM result: ${llmOutput.generations[0][0].text}\n`));
        // },
    }],
});

console.log(chalk.greenBright(result.output));
