# digital-power-case

#### The Twitter Sentiment Analysis product consists of a containerized data pipeline, Azure Storage Account and a Power BI dashboard
The end-product is the Power BI dashboard which provides insights into the Twitter Sentiment Analysis data.
The dataset behind this dashboard is linked to an Azure Blob Storage and retrieves the data from that Blob once a day (01 am Berlin time). \
The containerized data pipeline builds upon a Kaggle dataset: https://www.kaggle.com/kazanova/sentiment140.\
This dataset is to be made available in the 'data/twitter_sentiment/' folder with file name 'twitter_data.csv'.

_A later release of this data product would contain a dynamic retrieval of this dataset by means of the Kaggle API (as shown in "old/extract_data.py")._

Roadmap for data product is documented in __product_roadmap.txt__.

### Power BI report
The Power BI report is published in a Power BI workspace which requires access rights.
Access rights can be requested upon visiting the link below or can be set in advance in 
the Azure Active Directory of the organisational account that hosts the Embedded Power BI instance.

Available via:
https://app.powerbi.com/links/GzxVnj6fHN?ctid=2cb92ade-04ef-489b-aa1a-e5bbb5a0cc4c&pbi_source=linkShare



### Clone Github repo locally
First, clone the GitHub repository locally.

File structure:
- data
  - twitter_sentiment
    - twitter_data.csv
- model
  - lr_model_pipeline
  - lr_model_trained
- results
- main.py
- extract_data.py
- train_pipeline.py
- upload_to_blob.ppy
- requirements.txt
- Dockerfile 
- digital_power_case.pbix


### Install Docker desktop
Then, install Docker desktop via: 
    
    https://www.docker.com/products/docker-desktop

### Build Docker container & Run data pipeline
The Twitter Sentiment Analysis data pipeline runs in a Docker container. 
Build Docker image based on __Dockerfile__ in local repository: 

    docker build --rm -t <docker-image-name> .
    
    For example:
    docker build --rm -t brentvdwijdeven/pyspark-docker-dp . 

#### Please note, building the Docker Image also runs the file __main.py__ which runs the entire data pipeline.

### Run existing Docker container and command to run data pipeline


Then, depending on your operating system, run the docker image. 
The following example adds the --platform argument to enable the run command for Apple Macbook M1 chip.
The --entrypoint /bin/sh -itd argument keeps the container running in the background upon calling the 'run' argument.
This allows for interacting with the container Shell (which requires an actively running container)

    docker run --platform linux/amd64 --entrypoint /bin/sh -itd -v $(pwd):/job <docker-image-name>	

    For example:
    docker run --platform linux/amd64 --entrypoint /bin/sh -itd -v $(pwd):/job brentvdwijdeven/pyspark-docker-dp

Start existing container (requirement: run docker with --entrypoint /bin/sh -itd argument).
Then starting container also keeps it running!

### Start an existing Docker container

    docker start <containter_id>

Next, open Docker container CLI using Docker app UI. Go to 'Containers / Apps' and select the CLI button for your 
running container. This allows you to interact with the running docker container






