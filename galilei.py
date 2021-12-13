import streamlit as st


import numpy as np
import pandas as pd

import gensim
#from gensim.corpora import corpora
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import preprocess_documents
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models

#@st.cache
st.set_page_config(layout="wide")

lda = LdaModel.load("./models/first_model.model")
word2id = LdaModel.load("./models/first_model.model.id2word")

text = pd.read_csv("./data/abstract_lemmatized_processed_text.csv")

abstract_tokens = [simple_preprocess(x, deacc=True) for x in text["abstract_lemmatized_processed_text"]]
corpus = [word2id.doc2bow(x) for x in abstract_tokens]


topics = pyLDAvis.gensim_models.prepare(lda, corpus, word2id, mds="mmds", R=20)
pyLDAvis.save_html(topics, 'LDA_Visualization.html')

# html = """
# <div style ="background-color:#000000;padding:14px;border-radius:14px;">
# <h1 style ="color:white;text-align:center;font-size:56px;">
# UNDER CONSTRUCTION
# </h1>
# </div>
# """

#st.markdown(html, unsafe_allow_html = True)
st.title("Abstract Topics Under Construction")

page = st.selectbox("Select a page",("Topic Models", "Attention is All You Need"))

if page == "Topic Models":
    html_string = pyLDAvis.prepared_data_to_html(topics)
    st.components.v1.html(html_string, width=1300, height=800, scrolling=True)


if page == "Attention is All You Need":
    st.write("Under Construction")
