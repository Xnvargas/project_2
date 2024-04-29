# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Install Java and wget (required for Spark and downloading Spark)
RUN apt-get update && \
    apt-get install -y openjdk-11-jre-headless wget

# Set the JAVA_HOME environment variable
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64

# Download and install Spark
RUN wget https://downloads.apache.org/spark/spark-3.1.2/spark-3.1.2-bin-hadoop3.2.tgz && \
    tar -xzf spark-3.1.2-bin-hadoop3.2.tgz && \
    mv spark-3.1.2-bin-hadoop3.2 /spark && \
    rm spark-3.1.2-bin-hadoop3.2.tgz

# Set Spark related environment variables
ENV SPARK_HOME=/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
ENV PYSPARK_PYTHON=python3

# Install pyspark
RUN pip install pyspark==3.1.2

# If accessing S3, install Hadoop AWS package and AWS SDK
RUN pip install boto3
COPY hadoop-aws-3.2.0.jar /spark/jars/
COPY aws-java-sdk-bundle-1.11.375.jar /spark/jars/

# Copy the Python script into the container
COPY predict-5.py /predict-5.py

# Run the Python script
CMD ["spark-submit", "--packages", "org.apache.hadoop:hadoop-aws:3.2.0", "/predict-5.py", "--data_source", "s3://p2-datasets/ValidationDataset.csv", "--model_path", "s3://p2-datasets/myOutputFolder"]
