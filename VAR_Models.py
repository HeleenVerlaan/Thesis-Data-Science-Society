import logging
import pandas as pd
from statsmodels.tsa.api import VAR
from functions import causality_filter, train_test, dim_red, adfullertest, cross_validation, tune_hyperparameters, testing
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")
def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Loading datasets")
    df_topicslda = pd.read_csv('Lda_topics.csv', index_col=0) 
    df_topicslda = df_topicslda.set_index(pd.to_datetime(df_topicslda.index).strftime('%Y-%m'))
    df_topicshdbscan = pd.read_csv('Hdbscan_topics.csv', index_col=0)
    df_topicshdbscan = df_topicshdbscan.set_index(pd.to_datetime(df_topicshdbscan.index).strftime('%Y-%m'))
    df_topicskmeans = pd.read_csv('kmeans_topics.csv', index_col = 0)
    df_topicskmeans = df_topicskmeans.set_index(pd.to_datetime(df_topicskmeans.index).strftime('%Y-%m'))
    df_requests = pd.read_csv('Asylum_requests_and_family_reunification__nationality__sex_and_age_07052024_215402.csv', delimiter= ";")
    constant_columns = [col for col in df_topicslda.columns if df_topicslda[col].nunique() == 1]
    df_topicslda.drop(constant_columns, axis=1, inplace=True)
    print(f"Topics HDBSCAN: {len(df_topicshdbscan.columns)}\nTopics Kmeans: {len(df_topicskmeans.columns)}")
    logging.info("Preprocessing")
    df_requests = df_requests.transpose()
    df_requests = df_requests[:128]
    df_requests = df_requests.rename(columns = {0:'requests'})
    df_requests.index = [i.strip('*') for i in df_requests.index]
    df_requests.index = pd.to_datetime(df_requests.index).strftime('%Y-%m')
    logging.info("Concatenate datasets")
    df_lda= pd.concat([df_topicslda, df_requests],axis=1)
    df_hdbscan = pd.concat([df_topicshdbscan, df_requests], axis =1)
    df_hdbscan = df_hdbscan.drop('topic-1', axis =1)
    df_kmeans = pd.concat([df_topicskmeans, df_requests], axis =1)
    logging.info("Causality Filter")
    df_lda = causality_filter(df_lda)
    df_hdbscan = causality_filter(df_hdbscan)
    df_kmeans = causality_filter(df_kmeans)
    logging.info(f"Dataset shape of LDA: {df_lda.shape}\nData shape of HDBSCAN: {df_hdbscan.shape}\nDataset shape of K-means{df_kmeans.shape}")
    logging.info("PCA")
    df_lda = dim_red(df_lda, 'LDA')
    df_hdbscan = dim_red(df_hdbscan, 'HDBSCAN')
    df_kmeans = dim_red(df_kmeans, 'Kmeans')
    logging.info(f"Dataset shape of LDA: {df_lda.shape}\nData shape of HDBSCAN: {df_hdbscan.shape}\nDataset shape of K-means{df_kmeans.shape}")
    logging.info('Split into train and test sets')
    train_lda, test_lda = train_test(df_lda)
    train_hdbscan, test_hdbscan = train_test(df_hdbscan)
    train_kmeans, test_kmeans = train_test(df_kmeans)
    logging.info(f"Dataset shape of LDA: {train_lda.shape}\nData shape of HDBSCAN: {train_hdbscan.shape}\nDataset shape of K-means{train_kmeans.shape}")
    logging.info("Adfuller test on training data")
    d_lda, train_lda = adfullertest(train_lda)
    d_hdbscan, train_hdbscan = adfullertest(train_hdbscan)
    d_kmeans, train_kmeans = adfullertest(train_kmeans)
    logging.info(f"Dataset shape of LDA: {train_lda.shape}\nData shape of HDBSCAN: {train_hdbscan.shape}\nDataset shape of K-means{train_kmeans.shape}")
    logging.info("Expanding Window Cross Validation")
    logging.info("LDA")
    lags_lda, mae_lda, mape_lda= tune_hyperparameters(train_lda, VAR, d_lda, [1,2,3,4,5,6,7,8,9,10,11,12], [1,3,6])
    logging.info("BERTopic(HDBSCAN)")
    lags_hdbscan, mae_hdbscan, mape_hdbscan = tune_hyperparameters(train_hdbscan, VAR, d_hdbscan, [1,2,3,4,5,6,7,8,9,10,11,12], [1,3,6])
    logging.info("BERTopic(K-means)")
    lags_kmeans, mae_kmeans, mape_kmeans= tune_hyperparameters(train_lda, VAR, d_kmeans, [1,2,3,4,5,6,7,8,9,10,11,12], [1,3,6])
    logging.info(f"*MAE scores*\nLDA: {mae_lda}\nHDBSCAN: {mae_hdbscan}\nKmeans: {mae_kmeans}")
    logging.info(f"*MAPE scores*\nLDA: {mape_lda}\nHDBSCAN: {mape_hdbscan}\nKmeans: {mape_kmeans}")
    logging.info(f"Best Number of Lags\nLDA: {lags_lda}\nHDBSCAN: {lags_hdbscan}\nKmeans: {lags_kmeans}")
    logging.info("Train VAR Model")
    fc_lda, mae_lda, mape_lda = testing(train_lda, test_lda, VAR, lags_lda, 1, d_lda)
    print(fc_lda)
    print(test_lda['requests'])
    fc_hdbscan, mae_hdbscan, mape_hdbscan = testing(train_hdbscan, test_hdbscan, VAR, lags_hdbscan, 1, d_hdbscan)
    fc_kmeans, mae_kmeans, mape_kmeans = testing(train_kmeans, test_kmeans, VAR, lags_kmeans, 1, lags_kmeans)
    logging.info(f"\n - Results on Test data - \nModel       MAE           MAPE\n-------------------------------\nLDA       {mae_lda}       {mape_lda}\nHDBSCAN   {mae_hdbscan}       {mape_hdbscan}\nKmeans    {mae_kmeans}       {mape_kmeans}")
    for step in [3,6]:
        plt.plot(fc_lda[step], label = "LDA Forecast")
        plt.plot(fc_hdbscan[step], label = "BERTopic(HDBSCAN) Forecast")
        plt.plot(fc_kmeans[step], label = "BERTopic(K-means) Forecasts")
        plt.plot(test_lda['requests'].iloc[:step], label = "Actual Values")
        plt.xticks(rotation = 45)
        plt.legend()
        plt.savefig(f"Forcasts_{step}_steps")
        plt.clf()

main()