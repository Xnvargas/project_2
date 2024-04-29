from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.feature import VectorAssembler, Bucketizer
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import argparse

def load_and_predict(data_source, model_path):
    with SparkSession.builder.appName("Wine Quality Prediction Inference").getOrCreate() as spark:
        # Define the schema of the dataset
        schema = StructType([
            StructField("fixed acidity", DoubleType()),
            StructField("volatile acidity", DoubleType()),
            StructField("citric acid", DoubleType()),
            StructField("residual sugar", DoubleType()),
            StructField("chlorides", DoubleType()),
            StructField("free sulfur dioxide", DoubleType()),
            StructField("total sulfur dioxide", DoubleType()),
            StructField("density", DoubleType()),
            StructField("pH", DoubleType()),
            StructField("sulphates", DoubleType()),
            StructField("alcohol", DoubleType()),
            StructField("quality", DoubleType())
        ])

        # Load the trained Linear Regression model
        print("Loading model from:", model_path)
        model = LinearRegressionModel.load(model_path)

        # Load the validation dataset with a predefined schema
        print("Loading validation data from:", data_source)
        validation_df = spark.read.option("header", "true").option("delimiter", ";").schema(schema).csv(data_source)
        
        # Clean up column names by removing excessive quotation marks
        for col in validation_df.columns:
            new_col = col.replace('"', '').strip()
            validation_df = validation_df.withColumnRenamed(col, new_col)

        # List of feature columns (exclude 'quality' as it is the label)
        feature_columns = [col for col in validation_df.columns if col != 'quality']

        # Assemble features into a single vector column
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        validation_df = assembler.transform(validation_df)

        # Make predictions
        predictions = model.transform(validation_df)

        # Define bucketizer to convert 'prediction' into categories
        splits = [float('-inf'), 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, float('inf')]
        bucketizer = Bucketizer(splits=splits, inputCol="prediction", outputCol="predictedLabel")

        # Transform predictions into discrete classes
        predictions = bucketizer.transform(predictions)

        # Evaluate predictions using F1 score
        evaluator = MulticlassClassificationEvaluator(
            labelCol="quality",
            predictionCol="predictedLabel",
            metricName="f1")
        f1_score = evaluator.evaluate(predictions)
        print(f"F1 Score: {f1_score}")

        # Show predictions
        predictions.select("features", "prediction", "predictedLabel").show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Wine Quality')
    parser.add_argument('--data_source', required=True, help="The URI for your CSV wine validation data.")
    parser.add_argument('--model_path', required=True, help="The URI where the trained model is loaded from.")
    args = parser.parse_args()

    load_and_predict(args.data_source, args.model_path)
