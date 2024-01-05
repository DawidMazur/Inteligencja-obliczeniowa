# from langchain.llms import Ollama
#
# llm = Ollama(model="llama2:latest")
#
# from langchain.prompts import ChatPromptTemplate
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are world class technical documentation writer."),
#     ("user", "{input}")
# ])
#
#
# # llm.invoke("how can langsmith help with testing?")
#
# chain = prompt | llm
#
# res = chain.invoke({"input": "how can langsmith help with testing?"})
#
# print(res)
# # print response
# # print(chain.)
#

from dotenv import load_dotenv


from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.chat_models import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
load_dotenv()


tools = [TavilySearchResults(max_results=1)]

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-tools-agent")

# Choose the LLM that will drive the agent
# Only certain models support this
# llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
from langchain.llms import Ollama
llm = Ollama(model="llama2:latest")

# Construct the OpenAI Tools agent
agent = create_openai_tools_agent(llm, tools, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "what is LangChain?"})



