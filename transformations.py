from pyspark import SparkContext
from pyspark.sql import SparkSession
import pyspark.sql.types as tp
import timeit
import os

# prediction imports
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import  IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline, PipelineModel # ik begrijp het verschil hiertussen nog niet
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# determine script runtime
start_time = timeit.default_timer()


""" De initialize moet mogelijk naar main.py"""
# Initialize spark session
sc = SparkContext.getOrCreate()
spark = SparkSession.builder.master('local[*]').appName('twitter_sentiment_analysis').config("spark.ui.port","4040")\
    .getOrCreate()
# access spark web ui by visiting http://localhost:4040/jobs/ . This is only accessible while script is running!


def read_csv_to_spark_df(data_path, file_name):

    # schema to use for reading csv
    schema = tp.StructType([
        tp.StructField(name='target', dataType=tp.ByteType(), nullable=True),
        tp.StructField(name='ids', dataType=tp.IntegerType(), nullable=False),
        tp.StructField(name='date', dataType=tp.DateType(), nullable=True),
        tp.StructField(name='flag', dataType=tp.StringType(), nullable=True),
        tp.StructField(name='user', dataType=tp.StringType(), nullable=True),
        tp.StructField(name='text', dataType=tp.StringType(), nullable=True)
    ])

    # read the data again with the defined schema
    spark_df = spark.read.format("csv").option("header", "false", ).option("delimiter", ",").schema(schema) \
        .load(data_path + file_name)

    return spark_df


my_data = read_csv_to_spark_df('data/twitter_sentiment/', 'twitter_data.csv')


def drop_columns_and_rows(df, columns_to_drop): # can add type of df here in function argument

    # drop columns
    df = df.drop(*columns_to_drop)

    # drop rows with nan values
    df = df.dropna(how='any')

    return df


my_data = drop_columns_and_rows(df=my_data, columns_to_drop=['date', 'flag'])


def train_test_split(df, split_percentages):
    # split_percentages is list with three percentages: train, valid, test

    (train_set, val_set, test_set) = df.randomSplit(split_percentages, seed=2000)

    # hier nog ff testen of de dubbele () niet teveel is
    return (train_set, val_set, test_set)

# train test split


(train_set, val_set, test_set) = train_test_split(df=my_data , split_percentages=[0.95, 0.025, 0.025])
print(train_set.show())
print('hallee')



# some kind of prediction
"""
https://stackoverflow.com/questions/63703994/spark-v3-0-0-warn-dagscheduler-broadcasting-large-task-binary-with-size-xx
reducing task size => reduce the data its handling
First, check number of partitions in dataframe via df.rdd.getNumPartitions() After, increase partitions: df.repartition(100)
"""

# evt toevoegen: parameter tuning logistic regression, text cleaning, confusion matrix evaluation

# function: create pipeline and run pipeline


def save_pipeline(pipeline_model, path):
    # SAVE MODEL PIPELINE TO PATH

    pipeline_model.write().overwrite().save(path)

    return print('Pipeline saved to {}'.format(path))

def create_classification_pipeline():
    """ Deze functie met meer arguments opzetten zodat hij niet zo fixed is."""

    # ADDITIONAL FEATURE CREATION STEPS
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    cv = CountVectorizer(vocabSize=2 ** 16, inputCol="words", outputCol='cv')
    idf = IDF(inputCol='cv', outputCol="features", minDocFreq=5)  # minDocFreq: remove sparse terms
    label_stringIdx = StringIndexer(inputCol="target", outputCol="label")

    # Initialize Logistic Regression Step
    lr = LogisticRegression(maxIter=100)

    # stage_4 = LogisticRegression(featuresCol='features', labelCol='Runs') # hiervoor ook Vector Assembler gebruiken

    # Establish pipeline based on the steps defined before
    pipeline = Pipeline(stages=[tokenizer, cv, idf, label_stringIdx, lr])


    return pipeline





def fit_lr_model_pipeline(pipeline_model, pipeline_path, model_path, train_set):

    # fit pipeline to train dataset
    fit_pipeline = pipeline_model.fit(train_set)

    # transform train dataset with pipeline
    train_df_fitted = fit_pipeline.transform(train_set)

    # save pipeline
    save_pipeline(fit_pipeline, path=pipeline_path)
    # fit_pipeline.write().overwrite().save(pipeline_path)

    # save model
    lr_model = fit_pipeline.stages[4]
    print(lr_model)
    lr_model.write().overwrite().save(model_path)


    return fit_pipeline, train_df_fitted





def load_or_fit_model_pipeline(pipeline_path, model_path, train_set):

    if not os.path.exists('model/lr_model_pipeline'):

        pipeline = create_classification_pipeline()

        pipelineFit, train_df_fitted = fit_lr_model_pipeline(pipeline_model=pipeline,
                                                             pipeline_path=pipeline_path,
                                                             model_path=model_path, train_set=train_set)

    else:
        pipelineFit = PipelineModel.load(pipeline_path)

        train_df_fitted = pipelineFit.transform(train_set)

    return pipelineFit, train_df_fitted


pipeline_fitted, train_df_fitted = load_or_fit_model_pipeline(pipeline_path='model/lr_model_pipeline', model_path='model/lr_model_trained', train_set=train_set)





# for more universal saving and loading refer to https://stackoverflow.com/questions/57038445/load-model-pyspark
# WHEN TO USE A VECTOR ASSEMBLER TO CREATE A SINGLE COLUMN WITH ALL FEATURES?


def evaluate_pipeline_performance(test_set, pipeline_fitted):

    # transform validation dataset (or test dataset
    predictions_dataset = pipeline_fitted.transform(test_set)


    # evaluate function
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")

    # accuracy
    accuracy = predictions_dataset.filter(predictions_dataset.label == predictions_dataset.prediction).count() / float(val_set.count())

    # ROC AUC
    roc_auc = evaluator.evaluate(predictions_dataset)

    # nog geen idee wat dit doet
    # evaluator.evaluate(predictions_dataset)

    print("Accuracy Score: {0:.4f}".format(accuracy))
    print("ROC-AUC: {0:.4f}".format(roc_auc))

    return accuracy, roc_auc, predictions_dataset

accuracy, roc_auc, predictions_dataset = evaluate_pipeline_performance(test_set=test_set, pipeline_fitted=pipeline_fitted)

# save predictions as well
# predictions_dataset.toPandas().to_csv('results/predictions.csv') # er is niet veel heil in het opslaan van dit
end_time = timeit.default_timer()

print(end_time - start_time)



