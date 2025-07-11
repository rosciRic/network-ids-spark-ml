import time

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler,
    StandardScaler, Imputer
)
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr

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

TARGET = "Traffic Subtype"
#TARGET = 'Traffic Type'
#TARGET = 'Label'

TARGET_TO_DROP = {
    'Label': ['Traffic Type', 'Traffic Subtype'],
    'Traffic Type': ['Label', 'Traffic Subtype'],
    'Traffic Subtype': ['Label', 'Traffic Type']
}


def clean_data(spark):
    df = spark.read.csv("hdfs:///data/data.csv", header=True, inferSchema=True)
    initial_count = df.count()
    print(f"Dataset iniziale: {initial_count} righe")

    df = df.drop(*DROP_COLUMNS)
    df = df.drop(*TARGET_TO_DROP[TARGET])
    df = df.drop(*CONSTANT_FEATURES)
    df = df.drop(*HIGHLY_CORRELATED_FEATURES)

    df = df.dropDuplicates().na.drop()
    clean_count = df.count()
    print(f"Dataset pulito: {clean_count} righe")
    print(f"Feature mantenute: {len(df.columns) - 1}")

    return df


def preprocess(df):
    label_indexer = StringIndexer(inputCol=TARGET, outputCol="label").fit(df)
    df = label_indexer.transform(df)
    labels = label_indexer.labels

    feature_cols = [c for c in df.columns if c not in [TARGET, 'label']]
    numeric = [c for c, t in df.select(*feature_cols).dtypes if t in ['int', 'double']]
    categorical = [c for c in feature_cols if c not in numeric]

    imputer = Imputer(inputCols=numeric, outputCols=[f"{c}_imp" for c in numeric])
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in categorical]
    encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe") for c in categorical]

    assembled = [f"{c}_imp" for c in numeric] + [f"{c}_ohe" for c in categorical]
    assembler = VectorAssembler(inputCols=assembled, outputCol="features_unscaled")
    scaler = StandardScaler(inputCol="features_unscaled", outputCol="features")

    return df, [imputer] + indexers + encoders + [assembler, scaler], labels, assembled


def stratified_train_test_split(df, label_col=TARGET, train_ratio=0.8, seed=42):
    print("=== Esecuzione split stratificato ===")

    distinct_labels = [row[0] for row in df.select(label_col).distinct().collect()]

    fractions = {label: train_ratio for label in distinct_labels}

    train_df = df.sampleBy(label_col, fractions=fractions, seed=seed)

    joined_df = df.join(train_df, on=df.columns, how="leftanti")
    test_df = joined_df

    train_count = train_df.count()
    test_count = test_df.count()
    total = train_count + test_count

    print(f"=== Split completato ===")
    print(f"Train set: {train_count} righe ({train_count / total:.2%})")
    print(f"Test set: {test_count} righe ({test_count / total:.2%})")

    return train_df, test_df

def build_rf_model():
    return RandomForestClassifier(
        labelCol="label",
        featuresCol="features",
        numTrees=100,
        maxDepth=10,
        minInstancesPerNode=2,
        minInfoGain=0.0,
        seed=42
    )


def evaluate_model(predictions, labels):
    metrics = ["accuracy", "f1", "weightedPrecision", "weightedRecall"]
    for m in metrics:
        val = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName=m).evaluate(predictions)
        print(f"{m}: {val:.4f}")

    pred_lab = predictions.withColumn("label_name", expr("CASE " + " ".join(
        [f"WHEN label = {i} THEN '{l}'" for i, l in enumerate(labels)]) + " END")) \
        .withColumn("prediction_name", expr("CASE " + " ".join(
        [f"WHEN prediction = {i} THEN '{l}'" for i, l in enumerate(labels)]) + " END"))

    confusion = pred_lab.groupBy("label_name").pivot("prediction_name").count().fillna(0)
    confusion.write.mode("overwrite").option("header", "true").csv("hdfs:///results/confusion_matrix_traffic_subtypes")



start = time.time()
spark = SparkSession.builder.appName("RandomForest Full Pipeline").getOrCreate()

df = clean_data(spark)

df, preprocessing_stages, labels, assembled = preprocess(df)

train, test = stratified_train_test_split(df, label_col="label", train_ratio=0.8, seed=42)

rf = build_rf_model()
pipeline = Pipeline(stages=preprocessing_stages + [rf])

print("\n=== Training del modello ===")
model = pipeline.fit(train)

rf_model = model.stages[-1]
print("=== Best RandomForest Parameters ===")
print(f"numTrees: {rf_model.getNumTrees}")
print(f"maxDepth: {rf_model.getMaxDepth()}")
print(f"minInstancesPerNode: {rf_model.getMinInstancesPerNode()}")
print(f"impurity: {rf_model.getImpurity()}")

fi = rf_model.featureImportances
feature_importance = [(name, fi[idx]) for idx, name in enumerate(assembled)]
feature_importance.sort(key=lambda x: x[1], reverse=True)

print("\n=== Feature Importances")
for name, importance in feature_importance:
    print(f"{name}: {importance:.4f}")

print("\n=== Valutazione del modello ===")
predictions = model.transform(test)
evaluate_model(predictions, labels)

print(f"\n Pipeline completata in {round(time.time() - start, 2)} secondi")
spark.stop()