import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import SparkSession

DROP_COLUMNS = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp']


spark = SparkSession.builder.appName("Feature Analysis TII-SSRC-23").getOrCreate()

df = spark.read.csv("hdfs:///data/data.csv", header=True, inferSchema=True)
df = df.drop(*DROP_COLUMNS)
df = df.dropDuplicates().na.drop()
print(f"🔹 Numero di righe: {df.count()}")

numeric_cols = [c for c, t in df.dtypes if t in ('int', 'double')]
categorical_cols = [c for c, t in df.dtypes if t == 'string']

print("\n Statistiche descrittive (numeriche):")
df.select(numeric_cols).describe().show()

print("\nCardinalità colonne categoriche:")
for c in categorical_cols:
   count = df.select(c).distinct().count()
   print(f"{c}: {count} valori unici")

print("\nCalcolo correlazione tra feature numeriche...")

assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
df_vector = assembler.transform(df.select(numeric_cols)).select("features")

corr_matrix = Correlation.corr(df_vector, "features", "pearson").head()[0]
corr_array = corr_matrix.toArray()

corr_df = pd.DataFrame(corr_array, columns=numeric_cols, index=numeric_cols)

spark_corr_df = spark.createDataFrame(corr_df.reset_index())
spark_corr_df.write.mode("overwrite").option("header", "true").csv("hdfs:///results/correlation_matrix")
