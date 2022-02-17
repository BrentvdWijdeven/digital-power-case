from pyspark import SparkContext
from pyspark.sql import SparkSession

# Initialize spark session
sc = SparkContext.getOrCreate()
spark = SparkSession.builder.master('local[*]').appName('write_to_azure').config("spark.ui.port","4040")\
    .getOrCreate()

sc.conf.set(
    "fs.azure.account.key.<storage-account-name>.blob.core.windows.net",
    "<your-storage-account-access-key>"
)
