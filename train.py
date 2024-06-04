"""This module incorpores all the training and storing of the model."""

import pymongo
import pandas as pd
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
import logging
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import time

logging.getLogger().setLevel(logging.INFO)

client_mongo = pymongo.MongoClient("mongodb+srv://alextrem:1234@cluster0.x1yh6no.mongodb.net/?retryWrites=true&w=majority")
sc = SparkContext()
spark = SparkSession(sc)


if __name__ == "__main__":
    init_time = time.time()
    logging.info("Initializing training module.")

    logging.info("Retrieving data from exploitation zone.")
    # Database Name
    db = client_mongo["exploitation"]
    
    # Collection Name
    col = db["predictiveKPI_salePrice"] # read income collection

    df = pd.DataFrame(list(col.find()))  # convert it to pandas dataframe
    df.drop(['_id'], axis=1, inplace=True)   # drop _id column
    df = spark.createDataFrame(df)   # convert it to pyspark
    logging.info("Succesfully retrieved data from exploitation zone.")

    logging.info("Beggining defining the Model Pipeline.")
    numerical_features = ['bathrooms', 'distance', 'floor', 'hasLift',
       'hasPlan', 'hasVideo', 'numPhotos', 'rooms', 'size', 'year',
       'nof_incidents', 'pop', 'RFD']
    
    categorical_features = ['exterior', 'neighborhood', 'propertyType', 'status']

    # Apply StringIndexer to categorical columns
    indexers = [StringIndexer(inputCol=column, outputCol=column + "_indexed").fit(df) for column in categorical_features]

    # Concat numerical and categorical into one single column called features
    indexed_features = numerical_features + [col + "_indexed" for col in categorical_features]
    assembler = VectorAssembler(inputCols=indexed_features, outputCol="features")

    # Define Random Forest Regressor
    rf = RandomForestRegressor(labelCol="price", featuresCol="features", numTrees=10, maxBins=53)

    # Chain indexers, assembler and forest in a Pipeline
    pipeline = Pipeline(stages=indexers + [assembler, rf])
    logging.info("Finished defining the Model Pipeline.")

    
    logging.info("Beggining splitting the data into train and test.")
    train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)
    logging.info("Successfully splitted the data into train and test.")

    logging.info("Beggining with hyperparameter tunning.")
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [10, 20]) \
        .addGrid(rf.maxDepth, [5, 10]) \
        .build()

    # Create the cross-validator
    cross_validator = CrossValidator(estimator=pipeline,
                            estimatorParamMaps=paramGrid,
                            evaluator=MulticlassClassificationEvaluator(labelCol="price", metricName="accuracy"),
                            numFolds=5, seed=42)

    # Train the model with the best hyperparameters
    cross_validator = cross_validator.fit(train_data)
    model = cross_validator.bestModel

    logging.info("Finished hyperparameter tunning.")

    # Print the best hyperparameters
    rf = model.stages[-1]._java_obj.parent()
    logging.info(f"Selected number of trees: {rf.getNumTrees()}")
    logging.info(f"Selected max depth: {rf.getMaxDepth()}")


    logging.info("Beggining performing predictions of the train set.")
    # Make predictions.
    predictions = model.transform(test_data)

    # Example
    predictions.select("prediction", "price", "features").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(
        labelCol="price", predictionCol="prediction", metricName="rmse")

    rmse = evaluator.evaluate(predictions)
    logging.info(f"Predictions were done successfully with a")
    logging.info(f"Root Mean Squared Error (RMSE) on test data = {rmse}")

    logging.info(f"Finished training module successfully in {time.time() - init_time} seconds.")

    ### NEXT STEPS ###
    # Save the model using the MLFlow library

    client_mongo.close()
    spark.stop()
