"""This module incorporates all the training and storing of the model."""

import pymongo
import pandas as pd
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
import logging
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor, DecisionTreeRegressor
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import time
import mlflow
import mlflow.spark
import mlflow.sklearn


logging.getLogger().setLevel(logging.INFO)

def create_spark_session():
    sc = SparkContext()
    spark = SparkSession(sc)
    return spark, sc

def create_mongo_client():
    client_mongo = pymongo.MongoClient("mongodb+srv://alextrem:1234@cluster0.x1yh6no.mongodb.net/?retryWrites=true&w=majority")
    return client_mongo

def retrieve_data(client):
    db = client["exploitation"]
    col = db["predictiveKPI_salePrice"]
    df = pd.DataFrame(list(col.find()))
    df.drop(['_id'], axis=1, inplace=True)
    return df

def load_mlflow_model(model_name, model_version, data):
    # Load the model from the Model Registry
    model = mlflow.spark.load_model(f"models:/{model_name}/{model_version}")
    # Predict
    preds = model.transform(data)
    preds.select("prediction", "price", "features").show(5)


def main():
    init_time = time.time()
    logging.info("Initializing training module.")

    client_mongo = create_mongo_client()
    spark, sc = create_spark_session()

    try:
        logging.info("Retrieving data from exploitation zone.")
        df = retrieve_data(client_mongo)
        df = spark.createDataFrame(df)
        logging.info("Successfully retrieved data from exploitation zone.")

        logging.info("Beginning defining the Model Pipeline.")
        numerical_features = ['bathrooms', 'distance', 'floor', 'hasLift', 'hasPlan', 'hasVideo', 'numPhotos', 'rooms', 'size', 'year', 'nof_incidents', 'pop', 'RFD']
        categorical_features = ['exterior', 'neighborhood', 'propertyType', 'status']

        indexers = [StringIndexer(inputCol=column, outputCol=column + "_indexed").fit(df) for column in categorical_features]
        indexed_features = numerical_features + [col + "_indexed" for col in categorical_features]
        assembler = VectorAssembler(inputCols=indexed_features, outputCol="features")

        rf = RandomForestRegressor(labelCol="price", featuresCol="features", numTrees=10, maxBins=53)
        dt = DecisionTreeRegressor(labelCol="price", featuresCol="features", maxBins=53)

        rf_pipeline = Pipeline(stages=indexers + [assembler, rf])
        dt_pipeline = Pipeline(stages=indexers + [assembler, dt])
        logging.info("Finished defining the Model Pipeline.")

        logging.info("Beginning splitting the data into train and test.")
        train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)
        logging.info("Successfully split the data into train and test.")

        logging.info("Beginning hyperparameter tuning.")
        paramGrid_rf = ParamGridBuilder().addGrid(rf.numTrees, [10, 20]).addGrid(rf.maxDepth, [5, 10]).build()
        paramGrid_dt = ParamGridBuilder().addGrid(dt.maxDepth, [5, 10]).build()

        cross_validator_rf = CrossValidator(estimator=rf_pipeline, estimatorParamMaps=paramGrid_rf, evaluator=RegressionEvaluator(labelCol="price", metricName="rmse"), numFolds=5, seed=42)
        cross_validator_dt = CrossValidator(estimator=dt_pipeline, estimatorParamMaps=paramGrid_dt, evaluator=RegressionEvaluator(labelCol="price", metricName="rmse"), numFolds=5, seed=42)

        cross_validator_rf = cross_validator_rf.fit(train_data)
        cross_validator_dt = cross_validator_dt.fit(train_data)

        model_rf = cross_validator_rf.bestModel
        model_dt = cross_validator_dt.bestModel
        logging.info("Finished hyperparameter tuning.")

        # Extract parameters from the best Random Forest model
        rf_best_model = model_rf.stages[-1]._java_obj.parent()
        logging.info(f"Random Forest - Selected number of trees: {rf_best_model.getNumTrees()}")
        logging.info(f"Random Forest - Selected max depth: {rf_best_model.getMaxDepth()}")
        rf_params = [rf_best_model.getNumTrees(), rf_best_model.getMaxDepth()]
        # Extract parameters from the best Decision Tree model
        dt_best_model = model_dt.stages[-1]._java_obj.parent()
        logging.info(f"Decision Tree - Selected max depth: {dt_best_model.getMaxDepth()}")

        logging.info("Evaluating models on the test set.")
        predictions_rf = model_rf.transform(test_data)
        predictions_dt = model_dt.transform(test_data)
        predictions_rf.select("prediction", "price", "features").show(5)
        predictions_dt.select("prediction", "price", "features").show(5)

        evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="rmse")
        rmse_rf = evaluator.evaluate(predictions_rf)
        rmse_dt = evaluator.evaluate(predictions_dt)

        logging.info(f"Random Forest RMSE (test data): {rmse_rf}")
        logging.info(f"Decision Tree RMSE (test data): {rmse_dt}")

        best_model = model_rf if rmse_rf < rmse_dt else model_dt
        best_model_name = "Random Forest" if rmse_rf < rmse_dt else "Decision Tree"
        logging.info(f"Selected best model: {best_model_name}")
        logging.info("Saving the best model.")

        logging.info(f"Finished training module successfully in {time.time() - init_time} seconds.")


        logging.info("Starting MLflow run.")
        with mlflow.start_run() as run:
            # Log Random Forest model parameters and metrics
            mlflow.log_param("rf_num_trees", rf_best_model.getNumTrees())
            mlflow.log_param("rf_max_depth", rf_best_model.getMaxDepth())
            mlflow.log_metric("rf_rmse", rmse_rf)
            logging.info("Store the model in MLflow.")
            # Save the model
            # try:
            #     mlflow.spark.log_model(model.stages[-1], "random_forest_model")
            # except Exception as e:
            #     logging.error(f"An error occurred during model logging: {e}")

            # Log Decision Tree model parameters and metrics
            mlflow.log_param("dt_max_depth", dt_best_model.getMaxDepth())
            mlflow.log_metric("dt_rmse", rmse_dt)

            # Log best model information
            mlflow.log_param("best_model", str(model_rf.stages[-1]))
            logging.info("Models and parameters logged to MLflow.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        client_mongo.close()
        sc.stop()

if __name__ == "__main__":
    main()
