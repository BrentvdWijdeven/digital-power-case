from extract_data import download_or_load_dataset
from train_pipeline import read_csv_to_spark_df, drop_columns_and_rows, train_test_split, load_or_fit_model_pipeline, \
    evaluate_pipeline_performance, spark
import timeit


# determine script runtime
start_time = timeit.default_timer()

data_destination_folder = 'data/twitter_sentiment/'
download_data_file_name = 'twitter_data.csv'
file_name = 'training_twitter_data.csv'
incoming_data_file_name = 'incoming_twitter_data.csv'


# deze functie nog verplaatsen nar train_pipeline.py, dan ook function arguments toevoegen
def train_evaluate_save_pipeline():  # function name is niet overtuigend, misschien wil ik teveel doen in 1 functie

    # call download and unzip function for twitter dataset
    download_or_load_dataset(destination_folder=data_destination_folder,
                             download_data_file_name=download_data_file_name,
                             file_name=file_name, incoming_data_file_name=incoming_data_file_name)

    # create spark dataframe from downloaded csv
    my_data = read_csv_to_spark_df(data_destination_folder, file_name)

    # drop columns and rows
    my_data = drop_columns_and_rows(df=my_data, columns_to_drop=['date', 'flag'])

    # train test split
    (train_set, val_set, test_set) = train_test_split(df=my_data, split_percentages=[0.95, 0.025, 0.025])

    # load model pipeline if it is saved otherwise fit model and save it
    pipeline_fitted, train_df_fitted = load_or_fit_model_pipeline(pipeline_path='model/lr_model_pipeline',
                                                                  model_path='model/lr_model_trained',
                                                                  train_set=train_set)

    accuracy, roc_auc, predictions_dataset = evaluate_pipeline_performance(test_set=test_set,
                                                                           pipeline_fitted=pipeline_fitted)

    return predictions_dataset, pipeline_fitted


# df_predictions, fitted_pipeline = train_evaluate_save_pipeline()


def clean_incoming_tweets(raw_new_tweets):

    # function from train_pipeline.py
    cleaned_tweets = drop_columns_and_rows(raw_new_tweets, columns_to_drop=['date', 'flag'])

    return cleaned_tweets


def incoming_tweets_prediction(pipeline, incoming_tweets):

    incoming_tweets_predictions = pipeline.transform(incoming_tweets)

    return incoming_tweets_predictions


def run_and_save_incoming_tweets_prediction(pipeline, raw_new_tweets):

    cleaned_tweets = clean_incoming_tweets(raw_new_tweets=raw_new_tweets)

    incoming_tweets_predictions = incoming_tweets_prediction(pipeline=pipeline, incoming_tweets=cleaned_tweets)

    return incoming_tweets_predictions

#
# incoming_tweets_data = read_csv_to_spark_df(data_destination_folder, incoming_data_file_name)
# incoming_tweets_data.show()
#
# tweet_predictions = run_and_save_incoming_tweets_prediction(pipeline=fitted_pipeline,
#                                                             raw_new_tweets=incoming_tweets_data)
#
# tweet_predictions.show()
#
# tweet_predictions.toPandas().to_csv('results/incoming_twitter_data_predictions.csv')


# from pyspark import SparkContext
# from pyspark.sql import SparkSession
#
# """ De initialize moet mogelijk naar main.py"""
# # Initialize spark session
# sc = SparkContext.getOrCreate()
# pyspark = SparkSession.builder.master('local[*]').appName('twitter_sentiment_analysis').config("spark.ui.port", "4040")\
#     .getOrCreate()
#
#
# pyspark.conf.set(
#     "fs.azure.account.key.digitalpowerstorage1.blob.core.windows.net",
#     "tL2YmYTqrfeGma1/DRaTH9RTmZNKKZLn6O6fDKI3JnmJGfW8Nb+F2zRJ8m7uPRF71l5O1BRb88SrDc1lw+Pb+A=="
# )
#
#
# spark_df = spark.read.format('csv').option('header', True).\
#     load("wasbs://dp-blob1@digitalpowerstorage1.blob.core.windows.net/incoming_twitter_data.csv")
#



# print(spark_df.show())

end_time = timeit.default_timer()

print(end_time - start_time)

# to write to blob storage without pyspark but with normal python: https://www.youtube.com/watch?v=enhJfb_6KYU
# https://stackoverflow.com/questions/62384733/using-azure-storage-blob-to-write-python-dataframe-as-csv-into-azure-blob
