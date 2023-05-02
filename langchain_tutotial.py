
## TEST: Simple Call
def test_simple_call():
	from langchain.llms import OpenAI
	llm = OpenAI(temperature=0.9)
	text = "What would be a good company name for a company that makes colorful socks?"
	print(text)
	print(llm(text))


## TEST: Prompt
def test_prompt():
	from langchain.prompts import PromptTemplate

	prompt = PromptTemplate(
	    input_variables=["product"],
	    template="What is a good name for a company that makes {product}?",
	)
	prompt_text = prompt.format(product="yummy noodles")
	print(prompt_text)
	print(llm(prompt_text))


## TEST: Chain
def test_chain():
	from langchain.llms import OpenAI
	from langchain.chains import LLMChain

	chain = LLMChain(llm=llm, prompt=prompt)
	result = chain.run(product="funny stories")
	print(result)


## TEST: Mixed
# Test Result No Memory.
def test_mixed():
	from langchain.llms import OpenAI
	from langchain.chains import LLMChain
	from langchain.prompts import PromptTemplate

	# OpenAI has Request Rate Limitation.
	llm = OpenAI(temperature=0.9, model_name="gpt-3.5-turbo")
	prompt = PromptTemplate(
	    input_variables=["cmd"],
	    template="{cmd}",
	)
	chain = LLMChain(llm=llm, prompt=prompt)

	next_command = input("> ")
	while (next_command != "stop"):
		print(chain.run(next_command).strip())
		next_command = input("> ")
	print("Conv Over.")


## TEST Embed with Additional Search
def test_agent_with_search():
	from langchain.agents import load_tools
	from langchain.agents import initialize_agent
	from langchain.agents import AgentType
	from langchain.llms import OpenAI
	# First, let's load the language model we're going to use to control the agent.
	llm = OpenAI(temperature=0)
	# Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
	tools = load_tools(["ddg-search", "llm-math"], llm=llm)
	# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
	agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
	# Now let's test it out!
	agent.run("When was Washington D.C. built? How many years til today?")


## TEST Memory
def test_conversation_and_homemade_memory_llm():
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


## Chat Test
## No Memory Here
def test_chat():
	from langchain.chat_models import ChatOpenAI
	from langchain.schema import (
	    AIMessage,
	    HumanMessage,
	    SystemMessage
	)

	chat = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
	# print(chat([HumanMessage(content="Translate this sentence from English to French. I love programming.")]))

	import re
	p = re.compile('(#SYS)?(,| )*(.*)')
	def print_help():
		print("Hint: type `exit` to end conv; type `#SYS` to input context; type `help` for hints; "+
			"chat mode for anything else.")
	def get_input():
		return input("> ")
	def is_sys_input(m):
		return m.group(1) is not None
	def get_sys_msg(m):
		sys_msg = m.group(3)
		return SystemMessage(content=sys_msg)
	def get_chat_msg(m):
		chat_msg = m.group(3)
		return HumanMessage(content=chat_msg)
	get_msg = lambda f, x: print(chat([f(x)]).content)

	print_help()
	raw_input = get_input()
	while(raw_input != "exit"):
		if (raw_input.strip() == ""):
			raw_input = get_input()
			continue
		if (raw_input == "help"):
			print_help()
		m = p.match(raw_input)
		if (is_sys_input(m)):
			get_msg(get_sys_msg, m)
		else:
			get_msg(get_chat_msg, m)
		raw_input = get_input()
	print("> Bye!")

	# -> AIMessage(content="J'aime programmer.", additional_kwargs={})

	# messages = [
	#     SystemMessage(content="You are a helpful assistant that translates English to French."),
	#     HumanMessage(content="Translate this sentence from English to French. I love programming.")
	# ]
	# print(chat(messages))
	# -> AIMessage(content="J'aime programmer.", additional_kwargs={})

	# batch_messages = [
	#     [
	#         SystemMessage(content="You are a helpful assistant that translates English to French."),
	#         HumanMessage(content="Translate this sentence from English to French. I love programming.")
	#     ],
	#     [
	#         SystemMessage(content="You are a helpful assistant that translates English to French."),
	#         HumanMessage(content="Translate this sentence from English to French. I love artificial intelligence.")
	#     ],
	# ]
	# result = chat.generate(batch_messages)
	# print(result)
	# -> LLMResult(generations=[[ChatGeneration(text="J'aime programmer.", generation_info=None,
	# message=AIMessage(content="J'aime programmer.", additional_kwargs={}))], [ChatGeneration(
	# text="J'aime l'intelligence artificielle.", generation_info=None, message=AIMessage(content=
	# "J'aime l'intelligence artificielle.", additional_kwargs={}))]], llm_output={'token_usage':
	# {'prompt_tokens': 71, 'completion_tokens': 18, 'total_tokens': 89}})

# test_chat()

def test_chat_with_prompt():
	from langchain.chat_models import ChatOpenAI
	from langchain.prompts.chat import (
	    ChatPromptTemplate,
	    SystemMessagePromptTemplate,
	    HumanMessagePromptTemplate,
	)

	chat = ChatOpenAI(temperature=0)

	template="You are a helpful assistant that translates {input_language} to {output_language}."
	system_message_prompt = SystemMessagePromptTemplate.from_template(template)
	human_template="{text}"
	human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

	chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

	# get a chat completion from the formatted messages
	chat(chat_prompt.format_prompt(input_language="English", output_language="French", text="I love programming.").to_messages())
	# -> AIMessage(content="J'aime programmer.", additional_kwargs={})

def test_chain_with_chat_bot():
	from langchain.chat_models import ChatOpenAI
	from langchain import LLMChain
	from langchain.prompts.chat import (
	    ChatPromptTemplate,
	    SystemMessagePromptTemplate,
	    HumanMessagePromptTemplate,
	)

	chat = ChatOpenAI(temperature=0)

	template="You are a helpful assistant that translates {input_language} to {output_language}."
	system_message_prompt = SystemMessagePromptTemplate.from_template(template)
	human_template="{text}"
	human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
	chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

	chain = LLMChain(llm=chat, prompt=chat_prompt)
	chain.run(input_language="English", output_language="French", text="I love programming.")
	# -> "J'aime programmer."

def test_agent_chain_chat_bot():
	from langchain.agents import load_tools
	from langchain.agents import initialize_agent
	from langchain.agents import AgentType
	from langchain.chat_models import ChatOpenAI
	from langchain.llms import OpenAI
	from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper

	chat = ChatOpenAI(temperature=0)
	llm = OpenAI(temperature=0)
	tools = load_tools(["ddg-search", "llm-math"], llm=llm)

	agent = initialize_agent(tools, chat, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
	agent.run("Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?")
test_agent_chain_chat_bot()

def test_chat_with_memory():
	from langchain.prompts import (
	    ChatPromptTemplate,
	    MessagesPlaceholder,
	    SystemMessagePromptTemplate,
	    HumanMessagePromptTemplate
	)
	from langchain.chains import ConversationChain
	from langchain.chat_models import ChatOpenAI
	from langchain.memory import ConversationBufferMemory

	prompt = ChatPromptTemplate.from_messages([
	    SystemMessagePromptTemplate.from_template(
			"The following is a friendly conversation between a human and an AI. "+
			"The AI is talkative and provides lots of specific details from its context. "+
			"If the AI does not know the answer to a question, it truthfully says it does not know."),
	    MessagesPlaceholder(variable_name="history"),
	    HumanMessagePromptTemplate.from_template("{input}")
	])

	llm = ChatOpenAI(temperature=0)
	memory = ConversationBufferMemory(return_messages=True)
	conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

	conversation.predict(input="Hi there!")
	# -> 'Hello! How can I assist you today?'


	conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
	# -> "That sounds like fun! I'm happy to chat with you. Is there anything specific you'd like to talk about?"

	conversation.predict(input="Tell me about yourself.")
	# -> "Sure! I am an AI language model created by OpenAI. I was trained on a large dataset of text from the internet,
	 # which allows me to understand and generate human-like language. I can answer questions, provide information, 
	 # and even have conversations like this one. Is there anything else you'd like to know about me?"
