"""Main program code for the agent"""

from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.core.query_engine import PandasQueryEngine

load_dotenv()

# Create the path for the population.csv
population_path = os.path.join("data", "population.csv")
# Dataframe we want to run the PandasQueryEngine on
population_df = pd.read_csv(population_path)

# Create an engine allowing interfacing with data using the PandasQueryEngine input ("df") equal to our dataframe
#  verbose=True is "Intermediate generated instructions", steps or commands produced as part of 
#   the process of converting a high-level query (often in natural language) into executable code
population_query_engine = PandasQueryEngine(df=population_df, verbose=True, synthesize_response=True)

# Test for checking that the population.csv is loaded correctly:
print(population_df.head())

response = population_query_engine.query(
    "What is the country with the highest population? What is the country with the lowest population? Give both the countries and their populations."
)

print(str(response))