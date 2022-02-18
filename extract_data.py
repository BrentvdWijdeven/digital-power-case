from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import os

# DOES NOT WORK ON MAC!!!
# import os
# os.environ["KAGGLE_USERNAME"] = "brentvdwijdeven"
# os.environ["KAGGLE_KEY"] = "<683beb99f461a29642db9fdd2f0ac185"

# BEST IS TO USE kaggle.json BUT THIS FILE SHOULD BE PLACED IN USER HOME DIRECTORY!

# so for now, use pycharm environment variables to solve the problem
def download_and_unzip_twitter_dataset(destination_folder, data_file_name):

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
        os.rename(destination_folder + zip_files[0], destination_folder + data_file_name)


    return print('Dataset downloaded & unzipped')

def download_or_load_dataset(destination_folder, data_file_name):
    """ Opzet is nu een one-time download en nog niet een daily api call of iets dergelijks. dus nog niet optimaal. """

    if not os.path.exists(destination_folder + data_file_name):

        # download and unzip twitter dataset
        download_and_unzip_twitter_dataset(destination_folder, data_file_name)

    else:
        print('dataset is already downloaded and is available in destination folder')




"""
Possible improvements:
 - replace kaggle api with twitter api. Call twitter api directly and extract data there on a daily/hourly/...
    basis. 
    Then, also find a better way to store this data 
     
"""

