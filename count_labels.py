from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("Count Labels on Clean Data") \
    .getOrCreate()

DROP_COLUMNS = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp']

CONSTANT_FEATURES = [
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Bytes/Bulk Avg",
    "Fwd Packet/Bulk Avg",
    "Fwd Bulk Rate Avg"
]

HIGHLY_CORRELATED_FEATURES = [
   "ACK Flag Count",
   "Active Max",
   "Active Min",
   "Average Packet Size",
   "Bwd Bytes/Bulk Avg",
   "Bwd Header Length",
   "Bwd IAT Max",
   "Bwd IAT Min",
   "Bwd IAT Std",
   "Bwd Packet Length Mean",
   "Bwd Packet Length Std",
   "Bwd Packet/Bulk Avg",
   "Bwd Segment Size Avg",
   "Flow IAT Max",
   "Flow IAT Mean",
   "Flow IAT Min",
   "Fwd Act Data Pkts",
   "Fwd Header Length",
   "Fwd IAT Max",
   "Fwd IAT Mean",
   "Fwd IAT Min",
   "Fwd IAT Total",
   "Fwd Packet Length Mean",
   "Fwd Packet Length Min",
   "Fwd Packets/s",
   "Fwd Segment Size Avg",
   "Idle Max",
   "Idle Mean",
   "Idle Min",
   "PSH Flag Count",
   "Packet Length Max",
   "Packet Length Mean",
   "Packet Length Variance",
   "Subflow Bwd Packets",
   "Subflow Fwd Packets",
   "URG Flag Count"
]

df = spark.read.csv("hdfs:///data/data.csv", header=True, inferSchema=True)
df = df.drop(*DROP_COLUMNS)
df = df.drop(*CONSTANT_FEATURES)
df = df.drop(*HIGHLY_CORRELATED_FEATURES)

df = df.dropDuplicates().na.drop()

label_columns = ['Traffic Subtype', 'Label', 'Traffic_Type']

for label in label_columns:
    print(f"\n--- Conteggio per {label} ---")
    df.groupBy(col(label)).count().orderBy("count", ascending=False).show(32)

spark.stop()
