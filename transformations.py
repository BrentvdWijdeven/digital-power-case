from pyspark import SparkContext
from pyspark.sql import SparkSession
import pyspark.sql.types as tp
import timeit


# prediction imports
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import  IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# determine script runtime
start_time = timeit.default_timer()

# Initialize spark session
sc = SparkContext.getOrCreate()
spark = SparkSession.builder.master('local[*]').appName('twitter_sentiment_analysis').config("spark.ui.port","4040")\
    .getOrCreate()
# access spark web ui by visiting http://localhost:4040/jobs/ . This is only accessible while script is running!


# define a schema
schema = tp.StructType([
    tp.StructField(name= 'target',      dataType= tp.ByteType(),   nullable= True),
    tp.StructField(name= 'ids',         dataType= tp.IntegerType(),    nullable= False),
    tp.StructField(name= 'date',        dataType= tp.DateType(),   nullable= True),
    tp.StructField(name= 'flag',        dataType= tp.StringType(), nullable= True),
    tp.StructField(name= 'user',        dataType= tp.StringType(),    nullable= True),
    tp.StructField(name= 'text',        dataType= tp.StringType(),   nullable= True)
])


# read the data again with the defined schema
my_data = spark.read.format("csv").option("header", "false", ).option("delimiter", ",").schema(schema)\
    .load('data/twitter_sentiment/twitter_data.csv')

# my_data = my_data.limit(1000)

my_data = my_data.drop('date', 'flag')

# drop rows with nan values
my_data = my_data.dropna(how='any')



# train test split
(train_set, val_set, test_set) = my_data.randomSplit([0.95, 0.025, 0.025], seed=2000)
print(train_set.show())
print('hallee')


# some kind of prediction
"""
https://stackoverflow.com/questions/63703994/spark-v3-0-0-warn-dagscheduler-broadcasting-large-task-binary-with-size-xx
reducing task size => reduce the data its handling
First, check number of partitions in dataframe via df.rdd.getNumPartitions() After, increase partitions: df.repartition(100)
"""

# evt toevoegen: parameter tuning logistic regression, text cleaning, confusion matrix evaluation

evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")


tokenizer = Tokenizer(inputCol="text", outputCol="words")
cv = CountVectorizer(vocabSize=2**16, inputCol="words", outputCol='cv')
idf = IDF(inputCol='cv', outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
label_stringIdx = StringIndexer(inputCol="target", outputCol="label")
lr = LogisticRegression(maxIter=100)
pipeline = Pipeline(stages=[tokenizer, cv, idf, label_stringIdx, lr])

pipelineFit = pipeline.fit(train_set)
predictions = pipelineFit.transform(val_set)


accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(val_set.count())
roc_auc = evaluator.evaluate(predictions)

evaluator.evaluate(predictions)

print("Accuracy Score: {0:.4f}".format(accuracy))
print("ROC-AUC: {0:.4f}".format(roc_auc))

end_time = timeit.default_timer()

print(end_time - start_time)

# import time
# time.sleep(300)

# SAVE OUTPUT DF AS CSV?

# df = predictions.to_pandas_on_spark()


# predictions.write.option("header", 'true').csv("results/predictions.csv")
predictions.toPandas().to_csv('results/predictions.csv')


