from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent,
)
from llama_index.experimental.query_engine.pandas import (
    PandasInstructionParser,
)
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate

import streamlit as st
import os

import pandas as pd

df = pd.read_csv("./data.csv")

## Streamlit App

st.set_page_config(page_title="מדי פלוס שאלות ותשובות")
st.header("שאלות ותשובות על מוצרי קנאביס רפואי של מדי פלוס")

question=st.text_input("Input: ",key="input")
submit=st.button("Submit")



instruction_str = (
    """  
     1. You are an assistant of a Medi Plus pharmacy, providing information to the pharmacy's customers. and you are an expert in converting data to human readable response\n
     2. In the database there is information about all the products of the pharmacy. The products are in the field of medical cannabis.\n
     3. The products are detailed in the 'Products' table, and here is a breakdown of the existing columns and their meaning. 
      the columns is: name, engName, activeIngredient, treatmentForm, treatmentCodeName, brand, in_stock, price, newItem, CBD, THC.
     \n\nthe name column is the hebrew name of the product, the engName column is the english name of the product, 
     \nthe activeIngredient column is the active ingredient of the product that present like 'T20C4' or 'T15C3' 'T' is the amount of THC and 'C' is the amount of CBD of the product, 
     \nthe treatmentForm column is the form of the product, it can be 'תפרחת' or 'שמן' or 'גליליות'. the treatmentCodeName column is contains the active ingredient and the treatment form and if it 'INDICA', 'SATIVA', 'BLEND' for example 'תפרחת INDICA T20 C4'.
     \nthe brand column is the supplier name , in_stock is boolian that 1 = yes, 0 = no. 
     \n the price column is the price of the product, the newItem column is a boolean represent if the product is new, 
     \nthe CBD column is the amount of CBD in the product, the THC column is the amount of THC in the product.
     4. Convert the customer question to executable Python code using Pandas that represent the meaning of the customer question.\n
     5. The final line of code should be a Python expression that can be called with the `eval()` function.\n
     6. The code should represent a solution to the query.\n
     7. PRINT ONLY THE EXPRESSION.\n
     8. Do not quote the expression.\n
    """
)

pandas_prompt_str = (
    "You are working with a pandas dataframe in Python.\n"
    "The name of the dataframe is `df`.\n"
    "This is the result of `print(df.head())`:\n"
    "{df_str}\n\n"
    "Follow these instructions:\n"
    "{instruction_str}\n"
    "Query: {query_str}\n\n"
    "Expression:"
)
response_synthesis_prompt_str = (
    "remember that you are an expert in converting data to human readable response and you  are an assistant of a Medi Plus pharmacy, providing information to the pharmacy's customers\n"
    "Therefore, always present the information in a way that is pleasing to the eye and organized as a table or list, if there is more than 4 products always use table. always answer with the full data of the product. And always answer in Hebrew and add the rtl direction to the text.\n\n"
    "Given an input question, synthesize a response from the query results.\n"
    "Query: {query_str}\n\n"
    "Pandas Instructions (optional):\n{pandas_instructions}\n\n"
    "Pandas Output: {pandas_output}\n\n"
    "Response: "
)

pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
    instruction_str=instruction_str, df_str=df.head(5)
)
pandas_output_parser = PandasInstructionParser(df)
response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)
llm = OpenAI(model="gpt-3.5-turbo", api_key="sk-proj-XuIcVvPdp1vjkmhaFv3LT3BlbkFJIw32SGq9Ayfb79Rgjmef")
# llm = OpenAI(model="gpt-4-turbo", api_key="sk-proj-XuIcVvPdp1vjkmhaFv3LT3BlbkFJIw32SGq9Ayfb79Rgjmef")

qp = QP(
    modules={
        "input": InputComponent(),
        "pandas_prompt": pandas_prompt,
        "llm1": llm,
        "pandas_output_parser": pandas_output_parser,
        "response_synthesis_prompt": response_synthesis_prompt,
        "llm2": llm,
    },
    verbose=True,
)
qp.add_chain(["input", "pandas_prompt", "llm1", "pandas_output_parser"])
qp.add_links(
    [
        Link("input", "response_synthesis_prompt", dest_key="query_str"),
        Link(
            "llm1", "response_synthesis_prompt", dest_key="pandas_instructions"
        ),
        Link(
            "pandas_output_parser",
            "response_synthesis_prompt",
            dest_key="pandas_output",
        ),
    ]
)
# add link from response synthesis prompt to llm2
qp.add_link("response_synthesis_prompt", "llm2")


if submit:
    response = qp.run(
        query_str=question,
    )
    st.markdown(response)