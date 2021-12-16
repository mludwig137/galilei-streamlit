import streamlit as st


import numpy as np
import pandas as pd

import gensim
#from gensim.corpora import corpora
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import preprocess_documents
from gensim.models import LdaModel, phrases

import pyLDAvis
import pyLDAvis.gensim_models

st.set_page_config(layout="wide")

# lda = LdaModel.load("./models/first_model.model")
# word2id = LdaModel.load("./models/first_model.model.id2word")
#
# text = pd.read_csv("./data/abstract_lemmatized_processed_text.csv")
#
# abstract_tokens = [simple_preprocess(x, deacc=True) for x in text["abstract_lemmatized_processed_text"]]
# corpus = [word2id.doc2bow(x) for x in abstract_tokens]
#
#
# topics = pyLDAvis.gensim_models.prepare(lda, corpus, word2id, mds="mmds", R=20)
# pyLDAvis.save_html(topics, 'LDA_Visualization.html')

# html = """
# <div style ="background-color:#000000;padding:14px;border-radius:14px;">
# <h1 style ="color:white;text-align:center;font-size:56px;">
# UNDER CONSTRUCTION
# </h1>
# </div>
# """

#st.markdown(html, unsafe_allow_html = True)
st.title("Abstract Topic Model")

page = st.selectbox("Select a page",("Topic Model Unigram", "Topic Model N-gram", "Try a custom model", "Attention is All You Need"))

if page == "Topic Model Unigram":
    st.header("Unigram Topic Model")

    # load unigram lda model
    lda = LdaModel.load("./models/first_model.model")
    word2id = LdaModel.load("./models/first_model.model.id2word")

    text = pd.read_csv("./data/abstract_lemmatized_processed_text.csv")

    abstract_tokens = [simple_preprocess(x, deacc=True) for x in text["abstract_lemmatized_processed_text"]]
    corpus = [word2id.doc2bow(x) for x in abstract_tokens]

    topics = pyLDAvis.gensim_models.prepare(lda, corpus, word2id, mds="mmds", R=20)
    pyLDAvis.save_html(topics, 'LDA_Visualization.html')

    # https://discuss.streamlit.io/t/showing-a-pyldavis-html/1296/5
    html_string = pyLDAvis.prepared_data_to_html(topics)
    st.components.v1.html(html_string, width=1300, height=800, scrolling=False)

    st.subheader("Topic Assignment")

    cols = st.columns(5)
    cols[0].write("Topic 1: Quantum Grab bag")
    cols[1].write("Topic 2: Quantum Grab bag 2")
    cols[2].write("Topic 3: Astrophysics")
    cols[3].write("Topic 4: CS")
    cols[4].write("Topic 5: Condensed Matter(Top.)")

    cols2 = st.columns(5)
    cols2[0].write("Topic 6: Photonics")
    cols2[1].write("Topic 7: Quantum Grab bag 2")
    cols2[2].write("Topic 8: GR")
    cols2[3].write("Topic 9: Not sure...")
    cols2[4].write("Topic 10: Stellar Astrophysics")

    cols3 = st.columns(5)
    cols3[0].write("Topic 11: Galactic/Astronomy")
    cols3[1].write("Topic 12: String Theory")
    cols3[2].write("Topic 13: High Energy(Theory)")
    cols3[3].write("Topic 14: graphene...")
    cols3[4].write("Topic 15: Condensed Matter(Theory)")

    cols4 = st.columns(5)
    cols4[0].write("Topic 16: Quantum Computing")
    cols4[1].write("Topic 17: Particle(Experimental)")
    cols4[2].write("Topic 18: Particel(Experimental)")
    cols4[3].write("Topic 19: GR... Kerr Metric")
    cols4[4].write("Topic 20: Chemistry/Biophysics")

if page =="Topic Model N-gram":
    st.header("N-gram Topic Model")

    # load n-gram lda model
    ngram_lda = LdaModel.load("./models/137_ngram_model.model")
    ngram2id = LdaModel.load("./models/137_ngram_model.model.id2word")

    text = pd.read_csv("./data/abstract_lemmatized_processed_text.csv")

    abstract_tokens = [simple_preprocess(x, deacc=True) for x in text["abstract_lemmatized_processed_text"]]
    bigram = phrases.Phrases(abstract_tokens, min_count=2, threshold=.25, max_vocab_size=27000)
    trigram_bigram = phrases.Phrases(bigram[abstract_tokens], min_count = 2, threshold=.25, max_vocab_size=27000)

    bi_tri_gram_tokens = phrases.Phraser(trigram_bigram)
    ngram_tokens = [bi_tri_gram_tokens[x] for x in abstract_tokens]

    #ngram2id = gensim.corpora.Dictionary(ngram_tokens)
    ngram_corpus = [ngram2id.doc2bow(x) for x in ngram_tokens]

    ngram_topics = pyLDAvis.gensim_models.prepare(ngram_lda, ngram_corpus, ngram2id, mds="mmds", R=30)
    pyLDAvis.save_html(ngram_topics, 'ngram_LDA_Visualization.html')

    # https://discuss.streamlit.io/t/showing-a-pyldavis-html/1296/5
    ngram_html_string = pyLDAvis.prepared_data_to_html(ngram_topics)
    st.components.v1.html(ngram_html_string, width=1300, height=800, scrolling=True)

if page == "Try a custom model":
    st.header("Run a custom model")
    st.subheader("Adjust hyperparameters")

    num_topics = st.slider("No. of Topics", min_value=2, max_value=40, value=10, step=1, format=None)
    num_words = st.slider("Display n salient terms.", min_value=10, max_value=40, value=20, step=1, format=None)

    alpha = st.slider("Alpha(to the left)", min_value=0.01, max_value=5.0, value=1.0, step=0.01, format=None)
    alpha_auto = st.checkbox("Set alpha automatically")

    eta = st.slider("Eta(to the left, to the left)", min_value=0.01, max_value=2.0, value=1.0, step=0.01, format=None)
    eta_auto = st.checkbox("Set eta automatically")

    if alpha_auto == True:
        alpha = "auto"
    if eta_auto == True:
        eta = "auto"

    if st.button("Run model(stress test pending)"):
        with st.spinner('This may take a while... or fail...'):
            text = pd.read_csv("./data/abstract_lemmatized_processed_text.csv")
            abstract_tokens = [simple_preprocess(x, deacc=True) for x in text["abstract_lemmatized_processed_text"]]

            word2id = gensim.corpora.Dictionary(abstract_tokens)
            corpus = [word2id.doc2bow(x) for x in abstract_tokens]

            lda_custom = LdaModel(corpus=corpus, num_topics=num_topics, id2word=word2id, chunksize=500, passes=20,
                   update_every=1, alpha=alpha, eta=eta, random_state=137)

            topics_custom = pyLDAvis.gensim_models.prepare(lda_custom, corpus, word2id, mds="mmds", R=num_words)
            # pyLDAvis.save_html(topics, 'LDA_Visualization_custom.html')

            # https://discuss.streamlit.io/t/showing-a-pyldavis-html/1296/5
            html_string_custom = pyLDAvis.prepared_data_to_html(topics_custom)
            st.components.v1.html(html_string_custom, width=1300, height=800, scrolling=True)
        st.success("Complete!")
        st.balloons()

if page == "Attention is All You Need":
    st.write("Under Construction")
