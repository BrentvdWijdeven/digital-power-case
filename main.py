from extract_data import download_and_unzip_twitter_dataset
from train_pipeline import read_csv_to_spark_df, drop_columns_and_rows, train_test_split, load_or_fit_model_pipeline, \
    evaluate_pipeline_performance
import timeit


# determine script runtime
start_time = timeit.default_timer()

data_destination_folder = 'data/twitter_sentiment/'
file_name = 'twitter_data2.csv'


# deze functie nog verplaatsen nar train_pipeline.py, dan ook function arguments toevoegen
def train_evaluate_save_pipeline():  # function name is niet overtuigend, misschien wil ik teveel doen in 1 functie

    # call download and unzip function for twitter dataset
    download_and_unzip_twitter_dataset(data_file_name=file_name, destination_folder=data_destination_folder)

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


df_predictions, fitted_pipeline = train_evaluate_save_pipeline()


def clean_incoming_tweets(raw_new_tweets):

    cleaned_tweets = raw_new_tweets

    return cleaned_tweets


def incoming_tweets_prediction(pipeline, incoming_tweets):

    incoming_tweets_predictions = pipeline.transform(incoming_tweets)

    return incoming_tweets_predictions


def run_and_save_incoming_tweets_prediction(pipeline):

    cleaned_tweets = clean_incoming_tweets()

    incoming_tweets_predictions = incoming_tweets_prediction(pipeline=pipeline, incoming_tweets=cleaned_tweets)

    return incoming_tweets_predictions


tweet_predictions = run_and_save_incoming_tweets_prediction(pipeline=fitted_pipeline)

tweet_predictions.show()




end_time = timeit.default_timer()

print(end_time - start_time)

