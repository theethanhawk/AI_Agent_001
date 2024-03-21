"""Main program code for the agent"""

from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.core.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str, context
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

load_dotenv()

# Create the path for the population.csv
population_path = os.path.join("data", "population.csv")
# Dataframe we want to run the PandasQueryEngine on
population_df = pd.read_csv(population_path)

# Create an engine allowing interfacing with data using the PandasQueryEngine input ("df") equal to our dataframe
#  verbose=True is "Intermediate generated instructions", steps or commands produced as part of 
#   the process of converting a high-level query (often in natural language) into executable code
population_query_engine = PandasQueryEngine(
    df=population_df, verbose=True, instruction_str=instruction_str
)
population_query_engine.update_prompts({"pandas_prompt": new_prompt})


### Tests for checking that the population.csv is loaded correctly:
# print(population_df.head())

# response = population_query_engine.query(
#     "What is the country with the highest population? What is the country with the lowest population? Give both the countries and their populations."
# )

# print(str(response))

# population_query_engine.query(
#     "What is the population of Russia? And China? And the United States? And Canada?"
# )
### End of tests code


tools = [
    note_engine,
    QueryEngineTool(
        query_engine=population_query_engine, 
        metadata=ToolMetadata(
            name="population_data",
            description="this gives information about world population and demographics",
    ))
]

llm = OpenAI(model="gpt-3.5-turbo-1106")
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)