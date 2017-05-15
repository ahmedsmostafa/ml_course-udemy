from pyspark import SparkContext
from pyspark.streaming import StreamingContext

sc = SparkContext(appName="simpleApp", master="local[2]")
print("spark context version is: " + sc.version)

ssc = StreamingContext(sc,1)
print("spark streaming context created")

from pyspark.sql import SparkSession
spark = SparkSession(sc)
print("spark session version is: " + spark.version)

lines = ssc.socketTextStream("localhost", 9999)

words = lines.flatMap(lambda line: line.split(" "))

pairs = words.map(lambda word: (word, 1))
wordCounts = pairs.reduceByKey(lambda x,y: x + y)

wordCounts.pprint()