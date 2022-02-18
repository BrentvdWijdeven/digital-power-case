from extract_data import download_and_unzip_twitter_dataset
from transformations import read_csv_to_spark_df, drop_columns_and_rows, train_test_split, load_or_fit_model_pipeline, evaluate_pipeline_performance
import timeit


# determine script runtime
start_time = timeit.default_timer()


data_destination_folder = 'data/twitter_sentiment/'
file_name = 'twitter_data2.csv'
#
# from pyspark import SparkContext
# from pyspark.sql import SparkSession
# """ De initialize moet mogelijk naar main.py"""
# # Initialize spark session
# # sc = SparkContext.getOrCreate()
# spark = SparkSession.builder.master('local[*]').appName('twitter_sentiment_analysis').config("spark.ui.port", "4040")\
#     .getOrCreate()
# # access spark web ui by visiting http://localhost:4040/jobs/ . This is only accessible while script is running!


# call download and unzip function for twitter dataset
download_and_unzip_twitter_dataset(data_file_name=file_name, destination_folder=data_destination_folder)

# create spark dataframe from downloaded csv
my_data = read_csv_to_spark_df(data_destination_folder, file_name)

# drop columns and rows
my_data = drop_columns_and_rows(df=my_data, columns_to_drop=['date', 'flag'])

# train test split
(train_set, val_set, test_set) = train_test_split(df=my_data , split_percentages=[0.95, 0.025, 0.025])
print(train_set.show())
print('hallee')

# load model pipeline if it is saved otherwise fit model and save it
pipeline_fitted, train_df_fitted = load_or_fit_model_pipeline(pipeline_path='model/lr_model_pipeline', model_path='model/lr_model_trained', train_set=train_set)


accuracy, roc_auc, predictions_dataset = evaluate_pipeline_performance(test_set=test_set, pipeline_fitted=pipeline_fitted)


end_time = timeit.default_timer()

print(end_time - start_time)

