import logging
import pandas as pd
from functions import preprocessing, umass, get_pred_df, lda_vis, bert_vis
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
from bertopic import BERTopic
from sklearn.cluster import KMeans
import pickle

def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Loading Articles")
    df_articles = pd.read_csv("dutch-news-articles.csv")
    df_articles['datetime'] = df_articles['datetime'].apply(lambda x : x[:7])
    i = df_articles[df_articles['datetime'] == '2013-01'].index
    df_articles = df_articles.loc[i[0]:].reset_index(drop=True)
    articles = df_articles['content']
    docs = []
    for i, doc in enumerate(articles):
        docs.append(articles[i])
    logging.info("Preprocessing articles")
   # processed_docs = preprocessing(docs)
    with open("preprocessed_docs_LDA.pkl", "rb") as f:
        processed_docs = pickle.load(f)
    docs_lda = processed_docs.copy()
    #docs_bertopic = processed_docs.copy()
    '''
    for i in range(len(docs_bertopic)):
        docs_bertopic[i] = " ".join(docs_bertopic[i])
    logging.info("Training BERTopic")
    hdbscan = BERTopic("Dutch")
    topicsh, probs = hdbscan.fit_transform(documents= docs_bertopic)
    topic_count = len(hdbscan.get_topic_info())-1
    cluster_model = KMeans(n_clusters = topic_count)
    kmeans = BERTopic("Dutch", hdbscan_model= cluster_model, nr_topics=topic_count)
    topicsk, probs = kmeans.fit_transform(documents = docs_bertopic)
    logging.info(f"Obtained Topic Count: {topic_count}")'''
    topic_count = 500
    logging.info("Training LDA")
    id2word = Dictionary(docs_lda)
    corpus_lda = [id2word.doc2bow(doc) for doc in docs_lda]
    lda = LdaModel(corpus=corpus_lda, num_topics=topic_count, id2word=id2word)

    logging.info("Umass Score")
    lda_coh_mod = CoherenceModel(model=lda, texts= docs_lda, dictionary= id2word, coherence="u_mass")
    lda_umass = lda_coh_mod.get_coherence()
    #hdbscan_umass = umass(hdbscan, topicsh, docs_bertopic)
    #kmeans_umass = umass(kmeans, topicsk, docs_bertopic)
    logging.info(f"Umass Scores\nLDA: {lda_umass}")
    #logging.info(f"*Umass Scores*\nLDA: {lda_umass}\nHDSCAN: {hdbscan_umass}\nKmeans: {kmeans_umass}")
    logging.info("Creating new datasets for prediction")
    lda_pred = get_pred_df(lda, docs_lda, df_articles, "LDA", corpus=corpus_lda)
    #hdbscan_pred = get_pred_df(hdbscan, docs_bertopic, df_articles, "BERT")
    #kmeans_pred = get_pred_df(kmeans, docs_bertopic, df_articles, "BERT")
    lda_pred.to_csv("Lda_topics.csv")
    #hdbscan_pred.to_csv("Hdbscan_topics.csv")
    #kmeans_pred.to_csv("Kmeans_topcs.csv")
    logging.info("Visualization")
    lda_vis(lda)
    #bert_vis(hdbscan, 'HDBSCAN')
    #bert_vis(kmeans, 'Kmeans')   
main()