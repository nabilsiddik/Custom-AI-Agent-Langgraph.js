import { MessagesAnnotation, StateGraph } from '@langchain/langgraph'
import readline from 'node:readline/promises'
import { ChatGroq } from "@langchain/groq";
import { ToolNode } from '@langchain/langgraph/prebuilt';
import { TavilySearch } from "@langchain/tavily"
import { MemorySaver } from '@langchain/langgraph';


const checkPointer = new MemorySaver()

// Tavily web search tool
const webSearchTool = new TavilySearch({
    maxResults: 3,
    topic: "general",
    // includeAnswer: false,
    // includeRawContent: false,
    // includeImages: false,
    // includeImageDescriptions: false,
    // searchDepth: "basic",
    // timeRange: "day",
    // includeDomains: [],
    // excludeDomains: [],
});

// Initialise the tool node
const tools = [webSearchTool]
const toolNode = new ToolNode(tools)

const threads = {};

async function callModel(state, thread_id = "1") {
    const history = threads[thread_id] || [];

    const llm = new ChatGroq({ model: "openai/gpt-oss-120b", temperature: 0 })
        .bindTools(tools);

    const allMessages = [...history, ...state.messages];

    const response = await llm.invoke(allMessages);

    // save response to history
    threads[thread_id] = [...allMessages, response];

    return { messages: [response] };
}

function shouldContinue(state) {
    // Put condition weather call a tool or  not
    const lastMessage = state.messages[state.messages.length - 1]

    if (lastMessage.tool_calls.length > 0) {
        return 'tools'
    }

    return '__end__'
}

// Build the graph
const workflow = new StateGraph(MessagesAnnotation)
    .addNode('agent', callModel)
    .addNode('tools', toolNode)
    .addEdge('__start__', 'agent')
    .addEdge('tools', 'agent')
    .addConditionalEdges('agent', shouldContinue)

// compine the graph
const app = workflow.compile({ checkPointer })

async function main() {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    })

    while (true) {
        const userInput = await rl.question('You:')
        if (userInput === '/bye') break

        const finalState = await app.invoke({
            messages: [{ role: 'human', content: userInput }]
        }, { configurable: { thread_id: "1" } })

        const lastMessage = finalState.messages[finalState.messages.length - 1]

        console.log('AI: ', lastMessage.content)
    }

    rl.close()
}

main()