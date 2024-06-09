import { ChatOllama } from '@langchain/community/chat_models/ollama';
import { OllamaFunctions } from '@langchain/community/experimental/chat_models/ollama_functions';
import { DynamicStructuredTool } from '@langchain/core/tools';
import { ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate } from '@langchain/core/prompts';
import { convertToOpenAIFunction } from '@langchain/core/utils/function_calling';
import { JsonOutputFunctionsParser } from '@langchain/core/output_parsers/openai_functions';
import { RunnableLambda, RunnablePick, RunnablePassthrough } from '@langchain/core/runnables';

import { z } from 'zod';

import { inspect } from 'node:util';
import _ from 'lodash';
import chalk from 'chalk';

const commonOptions = { temperature: 0, seed: 19740822, keepAlive: '15m' };
const commonOptionsJSON = { ...commonOptions, format: 'json' };
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

// llama3 llama3:8b-instruct-Q8_0 has 8k ctx
// Prompt parse: ~500 t/s; generation: ~40 t/s
const llama3_8bLLMChat = new ChatOllama({ model: 'llama3:8b-instruct-Q8_0', ...commonOptions8k });
const llama3_8bLLMJSON = new ChatOllama({ model: 'llama3:8b-instruct-Q8_0', ...commonOptions8kJSON });

// phi3:medium-128k-instruct-q8_0 has 128k ctx but we'll only use 64k
// Prompt parse: ~250 t/s; generation: ~20 t/s
const phi3_14bLLMChat = new ChatOllama({ model: 'phi3:medium-128k-instruct-q8_0', ...commonOptions64k });
const phi3_14bLLMJSON = new ChatOllama({ model: 'phi3:medium-128k-instruct-q8_0', ...commonOptions64kJSON });

// mixtral:8x7b-instruct-v0.1-q8_0 - 32k context
// Prompt parse: ~200 t/s; generation: ~20-25 t/s
const mixtral7BLLMChat = new ChatOllama({ model: 'mixtral:8x7b-instruct-v0.1-q8_0', ...commonOptions32k });
const mixtral7BLLMJSON = new ChatOllama({ model: 'mixtral:8x7b-instruct-v0.1-q8_0', ...commonOptions32kJSON });

// llama3 llama3:70b-instruct-q8_0 has 8k ctx
// Prompt parse: ~65 t/s; generation: ~4-6 t/s
const llama3_70bLLMChat = new ChatOllama({ model: 'llama3:70b-instruct-q8_0', ...commonOptions8k });
const llama3_70bLLMJSON = new ChatOllama({ model: 'llama3:70b-instruct-q8_0', ...commonOptions8kJSON });

// llama3-chatqa llama3-chatqa:70b-v1.5-q8_0 has 32k ctx
// It wants special format with system (context appended to system), then user
// Prompt parse: ~65 t/s; generation: ~4-6 t/s
const llama3chatQA_70bLLMChat = new ChatOllama({ model: 'llama3-chatqa:70b-v1.5-q8_0', ...commonOptions32k });
const llama3chatQA_70bLLMJSON = new ChatOllama({ model: 'llama3-chatqa:70b-v1.5-q8_0', ...commonOptions32kJSON });

// mixtral:8x22b-instruct-v0.1-q4_0 - 64k training context but ollama sets it to 2k, has special tokens for tools and shit
// Prompt parse: ~60-80 t/s; generation ~11-12 t/s
const mixtral22bLLMChat = new ChatOllama({ model: 'mixtral:8x22b-instruct-v0.1-q4_0', ...commonOptions64k });
const mixtral22bLLMJSON = new ChatOllama({ model: 'mixtral:8x22b-instruct-v0.1-q4_0', ...commonOptions64kJSON });

let functionsLLM = new OllamaFunctions({ llm: mistralLLMJSON });

const calculatorSchema = z.object({
    operation: z
        .enum(['add', 'subtract', 'multiply', 'divide'])
        .describe('The type of operation to execute.'),
    number1: z.number().describe('The first number to operate on.'),
    number2: z.number().describe('The second number to operate on.'),
});

const calculatorTool = new DynamicStructuredTool({
    name: 'calculator',
    description: 'Can perform mathematical operations.',
    schema: calculatorSchema,
    func: async ({ operation, number1, number2 }) => {
        // Functions must return strings
        if (operation === 'add') {
            return `${number1 + number2}`;
        } else if (operation === 'subtract') {
            return `${number1 - number2}`;
        } else if (operation === 'multiply') {
            return `${number1 * number2}`;
        } else if (operation === 'divide') {
            return `${number1 / number2}`;
        } else {
            throw new Error('Invalid operation.');
        }
    },
});

const multiplyTool = new DynamicStructuredTool({
    name:'multiply',
    description: 'Can multiply two numbers.',
    schema: z.object({
        a: z.number().describe('The first number to multiply.'),
        b: z.number().describe('The second number to multiply.'),
    }),
    func: async ({ a, b }) => {
        return `${a * b}`;
    },
});

const tools = [calculatorTool, multiplyTool];
const toolMap = Object.fromEntries(
    tools.map((tool) => [tool.name, tool])
);

functionsLLM = functionsLLM.bind({ functions: tools.map((t) => convertToOpenAIFunction(t)) });

const toolChain = (toolCall) => {
    const chosenTool = toolMap[toolCall.name];
    return chosenTool.invoke(toolCall.arguments);
};
const toolChainRunnable = new RunnableLambda({
    func: toolChain,
});


const prompt = ChatPromptTemplate.fromMessages([
    HumanMessagePromptTemplate.fromTemplate('{input}'),
]);

const chain = prompt.pipe(functionsLLM)
                    .pipe(new JsonOutputFunctionsParser({ argsOnly: false }))
                    .pipe(RunnablePassthrough.assign({ output: toolChainRunnable }));

console.log(await chain.invoke({ input: 'What is the most populous state in the US, and what is the mean population of each state in the US?' }, {
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
            if (thought) {
                console.log(chalk.whiteBright(`\n${thought}\n`));
            } else {
                console.log(chalk.whiteBright(`\n${action.log.trim()}`));
            }
        },
        handleLLMStart(llm, message, runId, parentRunId, extraParams, tags, metadata, runName) {
            console.log(chalk.yellowBright(`\nAsking LLM: ${message}`));
        },
        handleLLMEnd(llmOutput, runId, parentRunId, tags) {
            console.log(chalk.bgYellowBright(`\nLLM result: ${llmOutput.generations[0][0].text}\n`));
        },
    }],

}));
