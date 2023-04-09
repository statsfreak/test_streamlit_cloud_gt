import streamlit as st
import catboost

import json
import os
import pickle
import streamlit as st
import pandas as pd
from snowflake.snowpark import Session
import warnings
from typing import Dict


from matplotlib import container
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt
from wordcloud import WordCloud, STOPWORDS


from snowflake.snowpark import Session, Row
from snowflake.snowpark.functions import call_udf, col
from snowflake.snowpark import functions as f

from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode
import openai
import requests

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

connection_parameters = {
  "account": os.environ["account"],
  "user": os.environ["username"],
  "password": os.environ["password"],
  "role": "esg_hackathon_role",
  "warehouse": "hackathon_wh",
  "database": "HACKATHON_ESG_BASE",
  "schema": "analytics"
}

session = Session.builder.configs(connection_parameters).create()
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             #background-image: url("https://cdn.pixabay.com/photo/2016/10/05/03/36/blue-1716030_1280.jpg");
             background-image: linear-gradient(rgba(255, 255, 255, 0.5), rgba(255, 255, 255, 0.75)), url("https://cdn.pixabay.com/photo/2016/10/05/03/36/blue-1716030_1280.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 
