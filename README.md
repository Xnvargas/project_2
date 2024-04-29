# project_2

# Training the model with AWS EMR
- Upload github files to an S3 bucket
    - Training python file: project_2b.py
    - Training datasets: TrainingDataset.csv, ValidationDataset.csv
    - Bootstrap Bash File: bootBash.sh
    - Inference application: predict-5.py
- Setup an EMR cluster with the following configuration:
    - Spark Application configuration
    - 1 Primary Node, 3 Core Nodes, 0 Task Nodes
    - Bootstrap Action
        - Numpy must be installed on the core nodes before executing the training algorithm.
        To do this, add a bootstrap action, and point to the bootBash.sh file s3 URI you obtained after uploading them to an S3 bucket.
    - Step Creation
        - Add a step to the cluster configuration with the following data:
            - Name: example-spark-app-name
            - Application type: Spark
            - Application JAR path: click browse S3 -> navigate to your S3 URI for training python file: project_2b.py
                - Arguments:
                    - --data_source s3://p2-datasets/TrainingDataset.csv
                    - --model_output_uri s3://p2-datasets/myOutputFolder
            - Example CLI command: spark-submit --deploy-mode cluster s3://p2-datasets/project_2b.py --data_source s3://p2-datasets/TrainingDataset.csv --model_output_uri s3://p2-datasets/myOutputFolder
- S3 output
    - Based on the --model_output_uri passed into the training session, your model will be in that directory in the S3 bucket. 

# Model Invokation
- AWS EMR 
    - Setup a clone of the above cluster, utilizing only a single primary and core node. 
    - In the steps configuration, remove all steps and add a new step as follows:
        - Name: example-invokation-app
        - Application type: spark
        - Application JAR path: browse S3 for the model invokation script -> predict.py
            - Arguments:
                - --data_source s3://p2-datasets/ValidationDataset.csv
                - --model_path s3://p2-datasets/myOutputFolder 
        - Example CLI: spark-submit --deploy-mode cluster s3://p2-datasets/predict-5.py --data_source s3://p2-datasets/ValidationDataset.csv --model_path s3://p2-datasets/myOutputFolder
    - Output:
        Model output is printed to the console, to view the logs from the worker node, simply ssh tunnel to the primary node following the guide under the 'applications' tab within the cluster management page.
            - Under application UIs, go to Resource manager (while the proxy is enabled from above step, easiest browser is Firefox) in your browser, and once you find your job execution, open up the LOG and you will find the output from the model invokation script.
