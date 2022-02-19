from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import os
import pandas as pd


# DOES NOT WORK ON MAC!!!
# import os
# os.environ["KAGGLE_USERNAME"] = "brentvdwijdeven"
# os.environ["KAGGLE_KEY"] = "<683beb99f461a29642db9fdd2f0ac185"

# BEST IS TO USE kaggle.json BUT THIS FILE SHOULD BE PLACED IN USER HOME DIRECTORY!

# so for now, use pycharm environment variables to solve the problem
def download_and_unzip_twitter_dataset(destination_folder, download_data_file_name):

    # initialize Kaggle api
    api = KaggleApi()
    api.authenticate()

    # download zip file with data
    api.dataset_download_files('kazanova/sentiment140', path='./')

    with zipfile.ZipFile('sentiment140.zip', 'r') as zipref:
        zipref.extractall(destination_folder)

        # not a very fault proof method to pick first file in zipextract
        zip_files = zipref.namelist()

        # rename file name to twitter_data.csv
        os.rename(destination_folder + zip_files[0], destination_folder + download_data_file_name)


    return print('Dataset downloaded & unzipped')


def split_dataset_training_and_incoming_sets(destination_folder, download_data_file_name, file_name, incoming_data_file_name):

    # read twitter csv
    file_path = destination_folder + download_data_file_name
    twitter_data = pd.read_csv(filepath_or_buffer=file_path, sep=',', header=None, encoding='latin-1')

    train_set_size = len(twitter_data) - 10000

    # RANDOM SHUFFLE DATAFRAME TO ENSURE INCOMING TWEETS SET CONTAINS BOTH POSITIVE AND NEGATIVE TWEETS
    twitter_data = twitter_data.sample(frac=1).reset_index(drop=True)

    # add columns here (instead of with read csv) as index column needed to be dropped.. Room for improvement here
    twitter_data.columns = ['target', 'ids', 'date', 'flags', 'user', 'text']



    train_test_validate_set = twitter_data.iloc[:train_set_size, :].copy()
    incoming_tweets_set = twitter_data.iloc[train_set_size:, :].copy()

    # # adjust date column to appropriate datetime format
    # incoming_tweets_set['date'] = pd.to_datetime(incoming_tweets_set["date"])
    # incoming_tweets_set['date'] = pd.to_datetime(incoming_tweets_set["date"].dt.strftime("%d/%m/%y %H:%M"))

    print('Dataset is splitted in a training set to train & test and a incoming tweets set that is used for '
          'running the pipeline')

    # save incoming_tweets_set to csv to load it at a later stage
    # save as parquet might be faster, but with csv its easier to inspect and show df
    train_test_validate_set.to_csv(destination_folder + file_name, index=False, header=False)
    incoming_tweets_set.to_csv(destination_folder + incoming_data_file_name, index=False, header=False)

    return train_test_validate_set

def download_or_load_dataset(destination_folder, download_data_file_name, file_name, incoming_data_file_name):
    """ Opzet is nu een one-time download en nog niet een daily api call of iets dergelijks. dus nog niet optimaal. """

    if not os.path.exists(destination_folder + download_data_file_name):

        # download and unzip twitter dataset
        download_and_unzip_twitter_dataset(destination_folder, download_data_file_name)

        # create separate train and incoming dataset
        split_dataset_training_and_incoming_sets(destination_folder, download_data_file_name, file_name,
                                                 incoming_data_file_name)

    else:
        print('Dataset is already downloaded and is available in destination folder')


"""
Possible improvements:
 - replace kaggle api with twitter api. Call twitter api directly and extract data there on a daily/hourly/...
    basis. 
    Then, also find a better way to store this data 
     
"""

