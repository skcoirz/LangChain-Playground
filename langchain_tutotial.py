# from langchain.llms import OpenAI

## TEST: Simple Call
# llm = OpenAI(temperature=0.9)
# text = "What would be a good company name for a company that makes colorful socks?"
# print(text)
# print(llm(text))


## TEST: Prompt
# from langchain.prompts import PromptTemplate

# prompt = PromptTemplate(
#     input_variables=["product"],
#     template="What is a good name for a company that makes {product}?",
# )
# prompt_text = prompt.format(product="yummy noodles")
# print(prompt_text)
# print(llm(prompt_text))


## TEST: Chain
# from langchain.chains import LLMChain

# chain = LLMChain(llm=llm, prompt=prompt)
# result = chain.run(product="funny stories")
# print(result)


## TEST: Mixed
# Test Result No Memory.
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# # OpenAI has Request Rate Limitation.
# llm = OpenAI(temperature=0.9, model_name="gpt-3.5-turbo")
prompt = PromptTemplate(
    input_variables=["cmd"],
    template="{cmd}",
)
# chain = LLMChain(llm=llm, prompt=prompt)

# next_command = input("> ")
# while (next_command != "stop"):
# 	print(chain.run(next_command).strip())
# 	next_command = input("> ")
# print("Conv Over.")


## TEST Embed with Additional Search
# from langchain.agents import load_tools
# from langchain.agents import initialize_agent
# from langchain.agents import AgentType
# from langchain.llms import OpenAI

# # First, let's load the language model we're going to use to control the agent.
# llm = OpenAI(temperature=0)

# # Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
# tools = load_tools(["ddg-search", "llm-math"], llm=llm)


# # Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
# agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# # Now let's test it out!
# agent.run("When was Washington D.C. built? How many years til today?")


## TEST Memory
from langchain import OpenAI, ConversationChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")
llm = OpenAI(temperature=0)
conversation = ConversationChain(llm=llm, verbose=False)

# traditional prompt won't be able to support memory. a trick is to store temporary memory manually and pass
# in as prompt.
prompt = PromptTemplate(
    input_variables=["cmd"],
    template="{cmd}",
)
chain = LLMChain(llm=llm, prompt=prompt)

# succeeded, but the problem is the input length limitation. memory is limited.
history = ""
fake_mem_prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="Past Chats: {history}; New question: {input}",
)
fake_mem_chain = LLMChain(llm=llm, prompt=fake_mem_prompt)

next_command = input("> ")
while (next_command != "stop"):
	history += "Human: " + next_command + "\n"
	print("AI 1: ", conversation.predict(input=next_command).strip())
	# print("AI 2: ", chain.run(next_command).strip())
	ai_answer = fake_mem_chain.run(history=history, input=next_command).strip()
	history += "AI: " + ai_answer + "\n"
	print("AI 3: " + ai_answer)
	next_command = input("> ")
print("Conv Over.")
