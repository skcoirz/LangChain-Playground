# import os
# os.environ["OPENAI_API_KEY"] = "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"

from langchain.llms import OpenAI

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

