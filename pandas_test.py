from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import pandas as pd

df = pd.read_csv('titanic_train.csv')
agent = create_pandas_dataframe_agent(
    OpenAI(temperature=0.5),
    df,
    verbose=True
)

# agent.run("how many rows are there")
agent.run("how many people have more than 3 sibligngs")
# agent.run("whats the square root of the average age?")