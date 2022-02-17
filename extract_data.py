from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

# DOES NOT WORK ON MAC!!!
# import os
# os.environ["KAGGLE_USERNAME"] = "brentvdwijdeven"
# os.environ["KAGGLE_KEY"] = "<683beb99f461a29642db9fdd2f0ac185"

# BEST IS TO USE kaggle.json BUT THIS FILE SHOULD BE PLACED IN USER HOME DIRECTORY!

# so for now, use pycharm environment variables to solve the problem

api = KaggleApi()
api.authenticate()

# download and unzip twitter sentiment data
api.dataset_download_files('kazanova/sentiment140', path='./')
with zipfile.ZipFile('sentiment140.zip', 'r') as zipref:
    zipref.extractall('data/twitter_sentiment/')

