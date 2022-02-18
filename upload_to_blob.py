from azure.storage.blob import ContainerClient

def upload(files, connection_string, container_name):
    container_client = ContainerClient.from_connection_string(connection_string, container_name)
    print('Uploading files to blob storage...')

    for file in files:
        blob_client = container_client.get_blob_client(file.name)

        with open(file.path, 'rb') as data:
            blob_client.upload_blob(data)
            print(f"{file.name} uploaded to blob storage")
import os

def get_files(dir):
    with os.scandir(dir) as entries:
        for entry in entries:
            if entry.is_file() and not entry.name.startswith('.'):
                yield entry

source_folder = 'results/'

csvs = get_files(source_folder)
print(csvs)

connection_string='DefaultEndpointsProtocol=https;AccountName=digitalpowerstorage1;AccountKey=tL2YmYTqrfeGma1/DRaTH9RTmZNKKZLn6O6fDKI3JnmJGfW8Nb+F2zRJ8m7uPRF71l5O1BRb88SrDc1lw+Pb+A==;EndpointSuffix=core.windows.net'

upload(csvs, connection_string, "dp-blob1")
