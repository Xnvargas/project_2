from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

def train_wine_quality_model(data_source, model_output_uri):
    """
    Trains a linear regression model on the provided wine quality dataset and saves the model.
    :param data_source: The URI of the wine data CSV, such as 's3://your-bucket/TrainingDataset.csv'.
    :param model_output_uri: The URI where the trained model is saved, such as 's3://your-bucket/model'.
    """
    with SparkSession.builder.appName("Wine Quality Prediction").getOrCreate() as spark:
        # Define schema for the dataset
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
        
        # Load the wine quality CSV data with a schema
        wine_df = spark.read.option("header", "true").option("delimiter", ";").schema(schema).csv(data_source)
        
        # Assemble features into a single vector
        feature_columns = wine_df.columns[:-1]  # all columns except the last one
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        wine_df = assembler.transform(wine_df)
        
        # Select features and label
        final_data = wine_df.select("features", "quality").withColumnRenamed("quality", "label")
        
        # Split data into training and test sets
        train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=42)
        
        # Create and train the linear regression model
        lr = LinearRegression(featuresCol="features", labelCol="label", maxIter=10, regParam=0.3, elasticNetParam=0.8)
        model = lr.fit(train_data)
        
        # Evaluate the model on test data
        predictions = model.transform(test_data)
        evaluator = RegressionEvaluator(metricName="rmse")
        rmse = evaluator.evaluate(predictions)
        print("RMSE on Test Data: ", rmse)
        
        # Save the trained model to the specified URI
        model.save(model_output_uri)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source', help="The URI for your CSV wine data, like an S3 bucket location.")
    parser.add_argument('--model_output_uri', help="The URI where the trained model is saved, like an S3 bucket location.")
    args = parser.parse_args()
    train_wine_quality_model(args.data_source, args.model_output_uri)
