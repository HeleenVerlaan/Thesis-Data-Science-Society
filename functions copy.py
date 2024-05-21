from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np
import logging
from statsmodels.tsa.stattools import adfuller
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import pickle
import spacy
import unidecode
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel

def causality_filter (df):
    cor_topics=[]
    for c in df.columns:
        caus = grangercausalitytests(df[[c, 'requests']], maxlag=6, verbose=False)
        p_values = [caus[i+1][0]['ssr_chi2test'][1] for i in range(6)]
        min_pvalue = np.min(p_values)
        if min_pvalue <= 0.05:
            cor_topics.append(c)
    cor_topics.append('requests')
    columns = df.columns
    df = df[cor_topics]
    logging.info(f"{len(columns)-len(cor_topics)} topics are removed from the dataset \n{len(cor_topics)} are left for forecasting")
    return df

def train_test(df):
    nobs = 6
    train,test = df[0:-nobs], df[-nobs:]
    return train,test

from statsmodels.tsa.stattools import adfuller
import logging

def adfullertest(train):
    d = 0  # Initialize differencing counter
    for c in train.columns:  # Iterate through each column in the DataFrame
        adf = adfuller(train[c], autolag='AIC')  # Perform ADF test
        pvalue = adf[1]  # Extract p-value from the ADF test result
        while pvalue > 0.05:  # Continue differencing until series becomes stationary
            d += 1  # Increment differencing counter
            train = train.diff().dropna()  # Difference the series and drop NaN values
            logging.info(f"Differencing {d} times. New shape: {train.shape}")  # Log the new shape of the DataFrame
            adf = adfuller(train[c], autolag='AIC')  # Perform ADF test again
            pvalue = adf[1]  # Extract p-value from the new ADF test result
    logging.info(f"All series are now stationary. \nDifferentiated {d} times")  # Log the number of differencing steps
    return d, train  # Return the number of differencing steps and the differenced DataFrame

def dim_red(df, model):    
    df2 = df.drop('requests', axis = 1)
    correlation_matrix = df2.corr()
    pca_counter = 0
    pca_models = []
    pca_topics = []
    for i, c in enumerate(df2):
        topics = []
        if c not in correlation_matrix.index:
            continue
        for r,v in enumerate(correlation_matrix[c]):
            if v == 1:
                continue
            elif v > 0.5 or v < -0.5:
                topics.append(correlation_matrix.index[r])
        if not topics:
            continue
        else:
            topics.append(c)
        pca = PCA()
        pca.fit(df[topics])
        pca_models.append(pca)
        max_explained_variance_ratio = max(pca.explained_variance_ratio_)
        n_components = list(pca.explained_variance_ratio_).index(max_explained_variance_ratio) + 1
        pca = PCA(n_components=n_components)
        pcas = [f'pca{pca_counter + p}' for p in range(1, n_components+1)]
        pca_counter += n_components
        df[pcas] = pca.fit_transform(df[topics])
        df = df.drop(topics, axis=1)
        pca_topics.append(topics)
        correlation_matrix = correlation_matrix.drop(topics, axis=1).drop(topics, axis=0)
        n_pcas = len(pca_models)
    logging.info(f"Number of PCA's = {n_pcas}")
    n_rows = (n_pcas + 4) // 5  # Calculate the number of rows needed
    n_cols = min(n_pcas, 5)  # Maximum of 5 columns
    if n_rows == 1:
        fig, axes = plt.subplots(nrows=n_rows, ncols = n_cols, figsize=(20, 4 * n_rows))
        for i, pca in enumerate(pca_models):
            col = i % n_cols
            ax = axes[col]    
            ax.bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, alpha = 0.5, align = 'center', color = 'blue')
            ax.set_title(f"PCA {i+1}")
            if col == 0:
                ax.set_ylabel('Variance Explained')         
                ax.set_xlabel('Principal Component')  
        for j in range(n_pcas, n_rows * n_cols):
            col = j % n_cols
            axes[col].axis('off')     
    else:
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 4 * n_rows))
        for i, pca in enumerate(pca_models):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            ax.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.5, align='center', color='blue')
            ax.set_title(f'PCA {i + 1}')
            # Set y-axis label for the first subplot in each row
            if col == 0:
                ax.set_ylabel('Variance Explained')
    # Set x-axis label for the last subplot row
            if col == 0:
                ax.set_xlabel('Principal Component')

# Hide empty subplots
        for j in range(n_pcas, n_rows * n_cols):
            row = j // n_cols
            col = j % n_cols
            axes[row, col].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust subplot layout to accommodate figure title
    plt.savefig(f"PCA_plot{model}")
    plt.clf()
    return df



def preprocessing(docs):
    stop = stopwords.words('Dutch')
    with open('countries.pkl', 'rb') as f:
        countries = pickle.load(f)
    f.close()
    with open('punctuation.pkl', 'rb') as f:
        punctuation = pickle.load(f)
    f.close()
    processed_docs = docs.copy()
    nlp = spacy.load('nl_core_news_sm')
    for i in range(len(processed_docs)):
        doc = nlp(processed_docs[i])
        processed_docs[i] = [token.lemma_ for token in doc]
        processed_docs[i] = [unidecode.unidecode(word) for word in processed_docs[i]]
        processed_docs[i] = [word.lower() for word in processed_docs[i]]
        processed_docs[i] = [word for word in processed_docs[i] if word not in countries]
        processed_docs[i] = [word for word in processed_docs[i] if word not in stop]
        processed_docs[i] = [word for word in processed_docs [i] if word not in punctuation]
    return processed_docs

def umass(model, topics, docs):
    cleaned_docs = model._preprocess_text(docs)
    vectorizer = model.vectorizer_model
    tokenizer = vectorizer.build_tokenizer()
    tokens = [tokenizer(doc) for doc in cleaned_docs]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    topic_words = [[words for words,_ in model.get_topic(topic)]for topic in range(len(set(topics))-1)]
    coherence_model = CoherenceModel(topics=topic_words, texts=tokens, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    umass = coherence_model.get_coherence()
    return umass

def get_pred_df (model, docs, articles, modeltype, corpus=None):
    if modeltype == 'BERT':
        topics = model.get_topic_info()
        columns = []
        zero = np.zeros((len(docs), len(topics)))
        topic_distribution = model.get_document_info(docs)['Topic']
        for topic in range(-1, len(topics)-1):
            columns.append(f"topic{topic}")
        df = pd.DataFrame(data=zero, columns=columns)
        if model.get_topic_info()['Topic'][0] == -1:
            for i, doc in enumerate(topic_distribution):
                df.iloc[i, doc+1] += 1
        else:
            for i, doc in enumerate(topic_distribution):
                df.iloc[i, doc] += 1
        df_pred = pd.concat([articles['datetime'], df], axis=1)
        grouped = df_pred.groupby('datetime')
        df_pred = grouped.sum()
    if modeltype == 'LDA':
        topic_distribution = [model.get_document_topics(doc) for doc in corpus]
        topics = len(model.get_topics())
        columns = []
        for topic in range(1,topics+1):
            columns.append(f"topic{topic}")
        zero = np.zeros((len(docs), len(model.get_topics())))
        df = pd.DataFrame(data = zero, columns=columns)
        for i, doc in enumerate(topic_distribution):
            for prob in doc:
                df.iloc[i,prob[0]] +=1
        df_pred = pd.concat([articles['datetime'], df], axis=1)
        grouped = df_pred.groupby('datetime')
        df_pred = grouped.sum()

    return df_pred

def lda_vis(model):
    sign_topics = pd.DataFrame(columns= ['word1', 'word2', 'word3', 'word4', 'word5', 'prob1', 'prob2', 'prob3', 'prob4', 'prob5'])
    x = model.show_topics(num_topics= 10, num_words= 5)
    for i, topic in enumerate(x):
        topic_info = []
        topic = topic[1] + " "
        split = topic.split('+ ')
        for t in reversed(split):
            topic_info.append(t[7:-2])
        for t in reversed(split):
            topic_info.append(float(t[:5]))
        sign_topics.loc[i] = topic_info
    fig, axes = plt.subplots(5, 2, figsize=(10, 15))  
    colors = ['xkcd:light blue', 'xkcd:orange' , 'xkcd:lavender', 'xkcd:dark pink', 'xkcd:grass green', 'xkcd:light navy', 'xkcd:pinkish', 'xkcd:golden', 'xkcd:faded blue', 'xkcd:rouge']
    for i in range(10):  # Iterate over each row of the DataFrame
        words = [sign_topics.iloc[i, j] for j in range(5)]  # Extract words data for the current row
        probs = [sign_topics.iloc[i, j] for j in range(5, 10)]  # Extract probabilities data for the current row
        axes[i // 2, i % 2].barh(words, probs, color=colors[i])
        axes[i // 2, i % 2].set_title(f'Topic {i+1}')  # Set title for the subplot
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig('LDA_topics.png')

def bert_vis(model, cluster):
    barchart = model.visualize_barchart(top_n_topics=10)
    barchart.write_html(f"BERTopic_topics({cluster}).html")

def cross_validation(trainset, model, d, lags, steps, n_splits =5):
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size= steps)
    mae_list = []
    mape_list = []
    forecasts = []
    for (train_index,test_index) in tscv.split(trainset):
        train = trainset.iloc[train_index]
        test = trainset.iloc[test_index]
        test = integrate(test, d)
        var = model(train)
        res = var.fit(lags)
        fc = res.forecast(train.values[-lags:], steps=steps)
        df_forecast = pd.DataFrame(fc, index = test[:steps].index, columns=train.columns)
        df_forecast = integrate(df_forecast, d)
        mae = mean_absolute_error(test[:steps]['requests'], df_forecast['requests'])
        mape = mean_absolute_percentage_error(test[:steps]['requests'], df_forecast['requests'])
        mae_list.append(mae)
        mape_list.append(mape)
        fc_list = list(df_forecast.requests)
        for f in fc_list:
            forecasts.append(f)
    mae = np.mean(mae_list)
    mape = np.mean(mape_list)
    return mae, mape, forecasts



def tune_hyperparameters(trainset, model, d, lag_values, step_values, n_splits=5):
    lags = []
    aver_mae = {}
    aver_mape = {}
    for step in step_values:
        r = {}
        mapes = {}
        for lag in lag_values:
            mae, mape, forecast = cross_validation(trainset=trainset, model=model, d=d, lags=lag, steps=step)
            r[lag] = mae
            mapes[lag] = mape
        min_lag = min(r, key=r.get)
        lags.append(min_lag)
        av_mae = np.mean(list(r.values()))
        av_mape = np.mean(list(mapes.values()))
        aver_mae[step]= av_mae
        aver_mape[step] = av_mape
    med = int(np.mean(lags))
    return med, aver_mae, aver_mape

def integrate(df, d):
    while d > 0:
        df = df.cumsum()
        d -= 1
    return df

def testing(train,test, model, lags, steps, d):
    steps = [1,3,6]
    maes = {}
    mapes = {}
    forecasts = {}
    var = model(train)
    res = var.fit(lags)
    for step in steps:
        fc = res.forecast(train.values[-lags:], steps=step)
        df_forecast = pd.DataFrame(fc, index = test[:step].index, columns=train.columns)
        df_forecast = df_forecast['requests']
        df_forecast = integrate(df_forecast, d)
        mae = round(mean_absolute_error(test[:step]['requests'], df_forecast), 3)
        mape = round(mean_absolute_percentage_error(test[:step]['requests'], df_forecast), 3)
        maes[step] = mae
        mapes[step] = mape
        forecasts[step] = df_forecast
    return forecasts, maes, mapes