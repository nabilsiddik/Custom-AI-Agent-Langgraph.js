import { MessagesAnnotation, StateGraph } from '@langchain/langgraph'
import readline from 'node:readline/promises'
import { ChatGroq } from "@langchain/groq";
import { ToolNode } from '@langchain/langgraph/prebuilt';
import { TavilySearch } from "@langchain/tavily"

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

async function callModel(state) {
    console.log('Calling LLM')

    const llm = new ChatGroq({
        model: "openai/gpt-oss-120b",
        temperature: 0
    }).bindTools(tools)

    const response = await llm.invoke(state.messages)

    return {messages: [response]}
}

function shouldContinue(state){
    // Put condition weather call a tool or  not
    console.log('state', state)
    const lastMessage = state.messages[state.messages.length - 1]
    const isToolCallingMandatory = lastMessage.tool_calls.length > 0

    if(isToolCallingMandatory){
        return 'tools'
    }

    return '__end__'
}

// Build the graph
const workflow = new StateGraph(MessagesAnnotation)
    .addNode('agent', callModel)
    .addNode('tool', ToolNode)
    .addEdge('__start__', 'agent')
    .addConditionalEdges('agent', shouldContinue)
    .addEdge('agent', '__end__')

// compine the graph
const app = workflow.compile()

async function main() {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    })

    while (true) {
        const userInput = await rl.question('You:')
        if (userInput === '/bye') break

        const finalState = await app.invoke({
            messages: [{ role: 'user', content: userInput }]
        })

        const lastMessage = finalState.messages[finalState.messages.length - 1]

        console.log('AI: ', lastMessage.content)
    }

    rl.close()
}

main()