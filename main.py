from extract_data import download_and_unzip_twitter_dataset
from transformations import *


data_destination_folder = 'data/twitter_sentiment/'
file_name = 'twitter_data2.csv'

# call download and unzip function for twitter dataset
download_and_unzip_twitter_dataset(data_file_name=file_name, destination_folder=data_destination_folder)


read_csv_to_spark_df(data_destination_folder, file_name)