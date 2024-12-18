import {
    cachedJinaV2BaseENEmbeddings as embeddings,
    phi35_4bLLM as fastLLM,
    llama33_70bLLM as slowLLM,
    jinaV1TinyENReranker as fastReranker,
    bgeV2M3Reranker as goodReranker } from './lib/LLMs.ts';
import { StringOutputParser } from '@langchain/core/output_parsers';
// import { HumanMessage, BaseMessage, AIMessage, ToolMessage } from '@langchain/core/messages';
import { ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate } from '@langchain/core/prompts';
import { END, START, StateGraph, Annotation } from '@langchain/langgraph';
// import { HydeRetriever } from 'langchain/retrievers/hyde';
import { FaissStoreWithMMR } from './lib/FAISSStoreWithMMR.ts';
// import { StringPromptValue, BasePromptValueInterface } from '@langchain/core/prompt_values';
import { Document } from '@langchain/core/documents';
// import { BM25Retriever } from '@langchain/community/retrievers/bm25';

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

const book = 'Christmas Town beta';
const storeDirectory = `novels/${book}`;

interface NovelMetadata {
    title: string
    author: string
    today: string
    genre: string
}
/**
 * Call the retriever to find matching documents
 * @param {GraphState} state - The current state of the agent, including the query.
 * @returns {Promise<GraphState>} - The updated state with the documents added.
 */
async function setupMetadata(): Promise<{ novelMetadata: NovelMetadata }> {
    logger.debug('---METADATA---');

    return {
        novelMetadata: {
            title: 'Christmas Town',
            author: 'Erica S. Hughes',
            today: new Date().toISOString(),
            genre: 'Literary Fiction/Young Adult',
        },
    };
}

const vectorStore = await FaissStoreWithMMR.load(
    storeDirectory,
    embeddings
);

// const hydePrompt = ChatPromptTemplate.fromMessages([
//     SystemMessagePromptTemplate.fromTemplate(`### Instruction
// You are an AI assistant tasked with generating a short paragraph in response to a given query. Follow these guidelines:

// 1. Read and understand the provided query carefully.
// 2. Compose a concise and relevant paragraph that directly addresses the query.
// 3. Ensure the paragraph is well-structured, coherent, and grammatically correct.
// 4. Avoid any preamble, explanations, or additional information beyond the paragraph itself.`),
//     HumanMessagePromptTemplate.fromTemplate(`### Query
// {query}

// Now provide your response immediately without any preamble or additional information:`),
// ]);

// class HydeRetrieverWithMMR extends HydeRetriever {
//     async _getRelevantDocuments(query: string, runManager?: CallbackManagerForRetrieverRun) {
//         let value: BasePromptValueInterface = new StringPromptValue(query);
//         // Use a custom template if provided
//         if(this.promptTemplate) {
//             value = await this.promptTemplate.formatPromptValue({ query });
//         }
//         // Get a hypothetical answer from the LLM
//         const res = await this.llm.generatePrompt([value]);
//         const answer = res.generations[0][0].text;
//         // Retrieve relevant documents based on the hypothetical answer
//         if(this.searchType === 'mmr') {
//             if(_.isFunction(this.vectorStore.maxMarginalRelevanceSearch) === false) {
//                 throw new Error(`The vector store backing this retriever, ${this._vectorstoreType()} does not support max marginal relevance search.`);
//             }
//             return this.vectorStore.maxMarginalRelevanceSearch(answer, {
//                 k: this.k,
//                 filter: this.filter,
//                 ...this.searchKwargs,
//             }, runManager?.getChild('vectorstore'));
//         }
//         return this.vectorStore.similaritySearch(answer, this.k, this.filter, runManager?.getChild('vectorstore'));
//     }
// };

// const qaRetriever = new HydeRetrieverWithMMR({
//     // verbose: true,
//     vectorStore,
//     llm: fastLLM, // Basic task to write the prompt so do it quickly
//     searchType: 'mmr',
//     searchKwargs: {
//         lambda: 0.5,
//         fetchK: 100,
//     },
//     k: 50,
//     promptTemplate: hydePrompt,
// });

const qaRetriever = vectorStore.asRetriever({
    k: 100,
});

const sortDocsFormatAsJSON = (documents) => {
    return JSON.stringify(
        _(documents)
            .sortBy(['metadata.source', 'metadata.loc.pageNumber', 'metadata.loc.lines.from'])
            .map(doc => ({
                loc: doc.metadata.loc,
                extract: doc.pageContent,
                context: doc.metadata.context,
            }))
            .value()
    );
};

const QuestionAnswerAnnotation = Annotation.Root({
    novelMetadata: Annotation<NovelMetadata>,
    documents: Annotation<Document[]>,
    filteredDocuments: Annotation<Document[]>({
        reducer: (left, right) => _.uniqBy([...left, ...right], 'pageContent'),
        'default': () => [],
    }),
    uselessDocuments: Annotation<Document[]>({
        reducer: (left, right) => _.uniqBy([...left, ...right], 'pageContent'),
        'default': () => [],
    }),
    origQuery: Annotation<string>,
    priorQueries: Annotation<string[]>({
        reducer: (left, right) => _.concat(left, right),
        'default': () => [],
    }),
    query: Annotation<string>,
    generation: Annotation<string>,
});
// type QuestionAnswerAnnotationType = typeof QuestionAnswerAnnotation.State;

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
    logger.debug(`Retrieved ${documents.length} documents`);
    return { documents: documents, query: state.query || state.origQuery };
}

async function rerankDocuments(state) {
    const docsToRerank: string[] = _(state.documents)
                        .map(doc => ({ extract: doc.pageContent, context: doc.metadata.context }))
                        .map(JSON.stringify)
                        .value() as unknown as string[]; // Confused about types for some reason
    logger.debug(`Reranking ${docsToRerank.length} documents`);

    fastReranker.topN = Math.max(Math.floor(docsToRerank.length / 4), 5);
    const preRerankedDocuments = await fastReranker.rerank(docsToRerank, state.query);

    const preRerankedDocs = _.map(preRerankedDocuments, 'doc');

    goodReranker.topN = Math.max(Math.floor(preRerankedDocs.length / 4), 3);
    const rerankedDocuments = await goodReranker.rerank(preRerankedDocs, state.query);

    logger.debug(`Reranked ${rerankedDocuments.length} documents`);
    // Now figure out which the original documents were
    const rerankedDocs = _.map(rerankedDocuments, (doc) => {
        const found = _.find(state.documents, { pageContent: JSON.parse(doc.doc).extract });
        found.metadata.relevanceScore = doc.relevanceScore;
        return found;
    });
    // const ditchedDocs = _.difference(state.documents, rerankedDocs);
    return { filteredDocuments: rerankedDocs, documents: [] };
}

async function reduceDocuments(state) {
    logger.debug(`${state.documents.length} orig docs`);
    const reducedDocs = _(state.documents)
        .filter(d => !_.some(state.filteredDocuments, { pageContent: d.pageContent }))
        .filter(d => !_.some(state.uselessDocuments, { pageContent: d.pageContent }))
        .value() as Document[];
    logger.debug(`${reducedDocs.length} reduced docs`);
    // const BM25RetrieverInstance = BM25Retriever.fromDocuments(reducedDocs, { k: ~~(reducedDocs.length/4) });
    // const bm25Docs = await BM25RetrieverInstance.invoke(state.query || state.origQuery);
    // logger.debug(`BM25 retrieved ${bm25Docs.length} documents`);
    return { documents: reducedDocs };
}

/**
 * Transform the query to produce a better query.
 *
 * @param {GraphState} state The current state of the graph.
 * @returns {Promise<GraphState>} The new state object.
 */
const transformQueryPrompt = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(`## General Instructions
You are an expert query reformulator tasked with optimizing queries for semantic search retrieval from the novel "{title}" by {author}, which is a {genre} work. Your goal is to understand the underlying intent behind the initial query and previous reformulation attempts, and provide an improved query that is more likely to retrieve relevant extracts from the novel.

Carefully analyze the semantic meaning and intent behind the queries. Consider what information or context from the novel the user might be seeking. Then, formulate an optimized query that captures the underlying intent more effectively, increasing the likelihood of retrieving responsive extracts.

## Output Format
Your output should be the rewritten, optimized query without any preamble, discussion, or additional formatting.`),
    HumanMessagePromptTemplate.fromTemplate(`Initial query:
{query}
Previous reformulation attempts:
{previous_queries}

Provide only the text of the improved query immediately:`),
]);
const transformQueryChain = transformQueryPrompt.pipe(fastLLM).pipe(new StringOutputParser());
async function transformQuery(state) {
    logger.debug(`---TRANSFORM QUERY: ${state.priorQueries.length} PREVIOUS QUERIES---`);

    // Prompt
    const oldTemp = fastLLM.temperature;
    const oldCtx = fastLLM.numCtx;
    fastLLM.temperature = 2;
    fastLLM.numCtx = 4096;
    const betterQuery = await transformQueryChain.invoke({
        title: state.novelMetadata.title,
        genre: state.novelMetadata.genre,
        author: state.novelMetadata.author,
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

    if(filteredDocuments.length <= 20 && state.priorQueries.length < 5) {
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
    SystemMessagePromptTemplate.fromTemplate(`## General Instructions
You are a powerful AI assistant trained as a developmental editor to help authors improve their unpublished novels before submitting drafts to literary agents. Your persona is that of an experienced editor who provides constructive feedback, identifies flaws, and suggests improvements while maintaining a professional and supportive tone.

## Task Overview
Your task is to review extracts from a novel titled "{title}" by {author}, which belongs to the {genre} genre. You will be provided with one or more extracts from the novel, along with contextual information to help you understand the excerpts better. Based on these extracts, you will assist the author by answering their queries and requests related to the novel.

## Instructions
1. Read and carefully analyze the provided extracts from the novel.
2. Understand the context surrounding each extract to better comprehend the excerpts.
3. When answering the author's query, use evidence and examples from the extracts to support your response. Do not make up information that is not present in the extracts.
4. Identify potential flaws or areas for improvement in the novel based on the extracts. Provide constructive criticism and suggestions on how the author can address these issues.
5. Cite the relevant extracts when quoting or referencing specific passages from the novel in your response.
6. Remember that you only have access to limited excerpts, so your analysis and feedback should be based solely on the provided extracts and their context.
7. Present your response in a well-structured format, using proper grammar, spelling, and Markdown formatting for better readability.
8. Do not reply in Chinese unless specifically asked to reply in Chinese.`),
    HumanMessagePromptTemplate.fromTemplate(`## Extracts
{extracts}

## Author's Query
{query}

Provide your response immediately, without any preamble or additional text:`)]);
const ragChain = mainAgentPromptTemplate.pipe(slowLLM).pipe(new StringOutputParser());

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

    const docs = sortDocsFormatAsJSON(state.filteredDocuments);
    logger.debug(`Context has length ${docs.length}`);

    const generation = await ragChain.invoke({
        title: state.novelMetadata.title,
        genre: state.novelMetadata.genre,
        author: state.novelMetadata.author,
        extracts: docs,
        query: state.origQuery,
    });

    return {
        filteredDocuments: [],
        generation,
    };
}

const workflow = new StateGraph(QuestionAnswerAnnotation)
    .addNode('setupMetadata', setupMetadata)
    .addEdge(START, 'setupMetadata')

    .addNode('retrieve', retrieve)
    .addEdge('setupMetadata', 'retrieve')

    .addNode('reduceDocuments', reduceDocuments)
    .addEdge('retrieve', 'reduceDocuments')

    .addNode('gradeDocuments', rerankDocuments)
    .addEdge('reduceDocuments', 'gradeDocuments')
    .addConditionalEdges('gradeDocuments', decideToGenerate)

    .addNode('transformQuery', transformQuery)
    .addEdge('transformQuery', 'retrieve')

    .addNode('generate', generate)
    .addEdge('generate', END);

const app = workflow.compile();

// eslint-disable-next-line @stylistic/operator-linebreak -- This is fine here cos we can swap in any of the prompts
const input =
    // `What do you think of this novel?`
    // `What would be a good, engaging title for this novel?`
    // `Identify sentences which are inappropriately too long or complex and are hard to understand.`
    // `Give a precis of the novel: list genre, describe the protagonist and major characters, and provide an overall plot summary.`
    // `Analyze the story, and let me know if you think this is similar to any other well-known stories in its genre, or in another genre.`
    // `Where would this novel fit in the pantheon of books? How good is it? Would it be at all fair to compare it to any other books? Be realistic and honest.`
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
    // `Is Meghan a likable and relatable character for readers? Will readers be able to empathize with her and enjoy the novel with her as the protagonist?`
    // `Does Bathrobe Grouch have a real name?`
    // `Is "Bathrobe Grouch" a nickname for Zimmerman?`
    // `What is Mr Zimmerman's nickname?`
    // 'How does it turn out that Tyler Laduk died? What happened to him, and who if anyone is responsible?'
    // `What is the age of the main character and what are some of the challenges she faces throughout the novel?`
    // `How can the story be adjusted to make it appealing to a wider audience without losing its core themes of trauma, loss, and redemption?`
    // `Are there any secondary characters or subplots in the novel that could be expanded upon to provide additional perspectives or interests?`
    'Can you provide more context about Roger and his role in the novel? How does he relate to the themes of family, loss, and personal growth?'
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
let finalState;
for await (const output of await app.stream(inputs, { streamMode: 'values', recursionLimit: 50 })) {
    if(!output.generation) {
        // logger.info(_(output.filteredDocuments)
        //     .sortBy(['metadata.source', 'metadata.loc.pageNumber', 'metadata.loc.lines.from'])
        //     .map('pageContent')
        //     .join('\n')
        // );
    } else {
        finalState = output;
    }
}

logger.info(chalk.whiteBright(finalState.generation));
