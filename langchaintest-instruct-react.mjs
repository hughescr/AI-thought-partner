import chalk from 'chalk';

import { OllamaEmbeddings } from '@langchain/community/embeddings/ollama';
import { Ollama } from '@langchain/community/llms/ollama';
import { FaissStore } from '@langchain/community/vectorstores/faiss';

import { RetrievalQAChain, loadQAMapReduceChain, loadQARefineChain, loadQAStuffChain } from 'langchain/chains';
import { DynamicTool } from '@langchain/core/tools';
import { WikipediaQueryRun } from "@langchain/community/tools/wikipedia_query_run";
import { ExaSearchResults } from "@langchain/exa";
import Exa from 'exa-js';

import { AgentExecutor } from 'langchain/agents';

import { PromptTemplate } from '@langchain/core/prompts';

import { RunnablePassthrough, RunnableSequence, } from "@langchain/core/runnables";
import { renderTextDescription } from "langchain/tools/render";
import { ReActSingleInputOutputParser } from "langchain/agents";
import { RunnableSingleActionAgent } from "langchain/agents";

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

const embeddings = new OllamaEmbeddings({ model: 'nomic-embed-text'});
// mistral-instruct-v0.2-2x7b-moe has 32k context, instruct model
const llm = new Ollama({ model: 'cas/mistral-instruct-v0.2-2x7b-moe', temperature: 0 });
// mix-crit is from mixtral:8x7b-instruct-v0.1-q3_K_L - 4k context, instruct model
// const llm = new Ollama({ model: 'mix-crit', temperature: 0 });
const storeDirectory = 'novels/Exile draft 2';
const vectorStore = await FaissStore.load(
    storeDirectory,
    embeddings
);

const [B_INST, E_INST] = ['<s>[INST]', '[/INST]</s>'];
const PROMPT = `You are a professional developmental editor, working for the author of a novel to help them improve drafts of their unpublished novel before they submit it to literary agents hoping to get published.

You can assume that the author is intimately familiar with the entire novel that they wrote.

You do not need to be overly fawning or laud the author if their work is not good; find the book's flaws when they exist, and help fix them.

Comprehensively answer the following question from the author as best you can with reference to the text where appropriate; when referencing the text, include extracts in your answer. Confirm all facts and assumptions before stating them.

You have access to some interns to help with researching your answer; interns are quite dumb and have no memory and do not remember what you've asked them to do before, they only know about the input that you send to them for that specific request. Treat interns like they're really dumb and you do not respect their abilities and you need to be very precise in your instructions or they will fuck it all up. These are the available interns:

{tools}

Use the following format to make use of interns:

\`\`\`
Thought: you should always think about what to do next in the way of any additional research
Action: only the name of the intern to use (should be one of [{tool_names}] verbatim with no escaping characters)
Action Input: the input to the intern
Observation: the output from the intern
\`\`\`

... (this Thought/Action/Action Input/Observation can repeat N times)

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

const qaStuffPromptTemplate = PromptTemplate.fromTemplate(`${B_INST}You are an intern who is follows directions as precisely as possible. Use the following extracts from the novel to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer. If the question is too complex for a solid answer, then ask for it to be broken down into simpler questions.

Question: {question}

Extracts:
{context}${E_INST}`);

const qaMapReduceMapPrompt = PromptTemplate.fromTemplate(`${B_INST}You are an intern whose job is to provide extracts and passages from a novel, working for a developmental editor.
Making no attempt to actually answer the question, check whether the extract is relevant to the question. If it is relevant, then return the extract verbatim.
If the question is too vague or complex, then ask for it to be broken down into simpler questions or to be more precise.

Question: {question}
Extract:
{context}${E_INST}`);

const qaMapReduceBasicReducePrompt = PromptTemplate.fromTemplate(`${B_INST}You are an intern who reads novels and provides simple answers about them, working for a developmental editor to help them identify parts of a novel that are relevant to their work.

Given the following extracted parts of the novel and a question, answer the question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer. You won't get in trouble for saying you don't know.
If the question is too vague or complex, then ask for it to be broken down into simpler questions or to be more precise.

Question: {question}
Extracts:
{summaries}${E_INST}`);
const qaMapReduceSummarizeReducePrompt = PromptTemplate.fromTemplate(`${B_INST}You are an intern whose job is to provide extracts and passages from a novel, working for a developmental editor.

Given the following portions of the novel and a question, return a verbatim extract or extracts from the novel that are relevant to the question.
If there are no relevant extracts, just say so. Don't try to make up an answer. You won't get in trouble for saying you don't know.
If the question is too vague or complex, then ask for it to be broken down into simpler questions or to be more precise.

Question: {question}
Extracts:
{summaries}${E_INST}`);

const qaRefineQuestionPrompt = PromptTemplate.fromTemplate(`${B_INST}You are a junior intern, working for a developmental editor. Answer the question, using only the context.

Given the following extracted parts of the novel and a question, answer the question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer. You won't get in trouble for saying you don't know.
If the question is too vague or complex, then ask for it to be broken down into simpler questions or to be more precise.

Question: {question}

Context:
{context}${E_INST}`);
const qaRefineRefinePrompt = PromptTemplate.fromTemplate(`${B_INST}You are a junior intern, working for a developmental editor.

Your boss the editor has asked the following question: {question}

If the question is too vague or complex, then ask for it to be broken down into simpler questions or to be more precise.

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
    retriever: vectorStore.asRetriever(40),
    // combineDocumentsChain: loadQAMapReduceChain(llm, {
    //     verbose: true,
    //     combineMapPrompt: qaMapReduceMapPrompt,
    //     combinePrompt: qaMapReduceBasicReducePrompt,
    // }),

    // combineDocumentsChain: loadQARefineChain(llm, {
    //     verbose: true,
    //     questionPrompt: qaRefineQuestionPrompt,
    //     refinePrompt: qaRefineRefinePrompt,
    // }),

    combineDocumentsChain: loadQAStuffChain(llm, {
        // verbose: true,
        prompt: qaStuffPromptTemplate,
    }),

    llm,
});

const extractRetrievalChain = new RetrievalQAChain({
    // verbose: true,
    retriever: vectorStore.asRetriever(20),

    combineDocumentsChain: loadQAMapReduceChain(llm, {
        // verbose: true,
        combineMapPrompt: qaMapReduceMapPrompt,
        combinePrompt: qaMapReduceSummarizeReducePrompt,
    }),

    // combineDocumentsChain: loadQARefineChain(llm, {
    //     verbose: true,
    //     questionPrompt: qaRefineQuestionPrompt,
    //     refinePrompt: qaRefineRefinePrompt,
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
        name: 'Joe',
        description: 'An untrustworthy junior intern who can only answer questions about what happens in this specific novel, based only on its text.',
        func: async (x) => (await qaChain.invoke({ query: x })).text,
    }),
    new DynamicTool({
        // verbose: true,
        name: 'Sally',
        description: 'An untrustworthy junior intern who can read the novel and get a verbatim extract or passage that demonstrates a relevant semantic concept.',
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

const result = await executor.invoke({
    input:
// `What do you think of this novel?`
// `Identify sentences which are too long or complex and are hard to understand.`
// `What would be a good, engaging title for this novel?`
// `Give a brief precis of the novel: list genre, describe the major characters, and provide an overall plot summary. Use the tools to be comprehensive.`
// `Analyze the story, and let me know if you think this is similar to any other well-known stories in its genre, or in another genre.`
// `Proofreading: are there any spelling, grammar, or punctuation errors that can distract readers from the story itself?`
// `Character development: Analyze the important characters and assess how well-developed they are, with distinct personalities, backgrounds, and motivations. This will make them more relatable and engaging to readers.`
// `Plot structure: Analyze whether the story's events are in a clear and coherent sequence, with rising action, climax, falling action, and resolution. This will help maintain reader interest throughout the novel.`
// `Subplots: Analyze the sub-plots and minor characters to verify that they add to the story instead of distracting from it. Sub-plots and side-characters should enhance the story and not confuse the reader. Point out any flaws.`
// `Show, don't tell: Analyze whether the story simply tells readers what is happening or how characters feel, or whether it uses vivid descriptions and actions to show them. This will make the writing more engaging and immersive.`
// `Consistent point of view: Does the novel stick to one consistent point of view throughout, whether it be first person, third person limited, or omniscient? This will help maintain a cohesive narrative voice.`
// `Active voice: Does the writing use active voice instead of passive voice whenever possible? This makes the writing more direct and engaging.`
// `Vary sentence structure: Does the writing break up long sentences with shorter ones to create rhythm and variety? This will make the writing more dynamic and interesting to read.`
// `Analyze the story from the point of view of a potential reader who purchases the book.`
`Analyze the story from the point of view of a literary agent reading this book for the first time and trying to decide if they want to represent this author to publishers.`
// `Provide suggestions on how to improve any confusing parts of the plot. If there are other narrative elements which should be revised and improved, point them out.`
// `List all the chapters in the book, and give a one-sentence summary of each chapter.`
// `What do you dislike the most about the book? What needs fixing most urgently?`
// `Who is the ideal audience for this book? What will they enjoy about it? What might they dislike about it? How can the story be adjusted to make it appear to a wider audience?`
// `Identify any repetitive or superfluous elements in the book.`
// `Identify any subplots which don't lead anywhere and just distract from the main story.`
},
{
    callbacks: [{
        handleToolStart(tool, input, runId, parentRunId, tags, metadata, runName) {
            console.log(chalk.blueBright(`Boss asks intern ${runName}: ${input}`));
        },
        handleToolEnd(output, runId, parentRunId, tags) {
            console.log(chalk.blue(`Intern responds: ${output}`));
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
