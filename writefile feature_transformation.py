"""
title: Feature transformation
description: Recast RDD as pandas.Dataframe and pipelines several operations to create numerical features for downstream ML.
"""

# Standard Library Imports
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, List, Tuple, Dict

# Third-Party Imports
from dotenv import load_dotenv
import numpy as np

# PySpark Imports
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor
from pyspark.ml import Model
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
from pyspark.sql.functions import log

# Local imports
from main_provider_curated_job_listings_rdd import provide_curated_job_listings_rdd
from preprocess import standardize_text_job_listings
from phrases import read_significant_features_from_json
from features import prepare_wages_and_features

# Set up logging for debug purposes
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Environment setup
load_dotenv()

# Pipeline parameters
MINDF = int(os.getenv("MINDF", 10))  # Default to 10 if not set
BINARY = os.getenv("BINARY", "True").lower() == "true"  # Default to True

#
# (0) recast listing as (distributed) spark dataframe
#

def validate_and_create_dataframe(
    spark: SparkSession,
    rdd: Any,
    validity_threshold: float = 0.8
) -> DataFrame:
    """
    Validates an RDD containing (wage, ngrams) pairs and converts it into a Spark DataFrame.

    Parameters:
        spark (SparkSession): The active Spark session.
        rdd (RDD): The input RDD containing (wage, ngrams) pairs.
        validity_threshold (float): The minimum fraction of valid entries required (default: 0.8).

    Returns:
        DataFrame: A Spark DataFrame with valid (wage, ngrams) pairs.
    """
    total_entries = rdd.count()

    # Define validation function
    def is_valid(entry: Tuple[Any, Any]) -> bool:
        if not isinstance(entry, tuple) or len(entry) != 2:
            return False
        wage, ngrams = entry
        if wage is not None and not isinstance(wage, (int, float)):
            return False
        if not isinstance(ngrams, list) or not all(isinstance(token, str) for token in ngrams):
            return False
        return True

    # Filter valid entries
    valid_rdd = rdd.filter(is_valid)
    valid_count = valid_rdd.count()
    valid_fraction = valid_count / total_entries if total_entries > 0 else 0

    logging.info(f"Total entries: {total_entries}")
    logging.info(f"Valid entries: {valid_count} ({valid_fraction:.2%})")

    if valid_fraction < validity_threshold:
        raise ValueError(f"Valid data percentage ({valid_fraction:.2%}) below threshold ({validity_threshold:.2%}). Aborting.")

    # Create DataFrame from valid entries
    df = spark.createDataFrame(valid_rdd, ["wage", "ngrams"])

    # Count and log None values in wage
    none_wage_count = df.filter(df.wage.isNull()).count()
    if none_wage_count > 0:
        logging.warning(f"Entries with None wages: {none_wage_count}")

    # Log the number of non-labeled (None) instances at warn level
    logging.warning(f"Number of non-labeled (None) wage instances: {none_wage_count}")

    # Filter out None wage entries
    df = df.filter(df.wage.isNotNull())

    # Compute the log scale of the wage
    df = df.withColumn("log_wage", log("wage"))

    return df

#
# (1) vectorize and provide feature names
#

def vectorize_dataframe(df: DataFrame, min_df: int = MINDF, binary: bool = BINARY) -> DataFrame:
    """
    Transforms a DataFrame by applying CountVectorizer on the 'ngrams' column.

    Parameters:
        df (DataFrame): The input DataFrame with 'ngrams' column.
        min_df (int, optional): Minimum document frequency for tokens. Defaults to value from .env.
        binary (bool, optional): Whether to use binary representation. Defaults to value from .env.

    Returns:
        DataFrame: A vectorized DataFrame with a new 'features' column.
    """
    # Load parameters from environment variables if not provided
    if min_df is None:
        min_df = int(os.getenv("MIN_DF", 10))  # Default to 10 if not set
    if binary is None:
        binary = os.getenv("BINARY", "True").lower() == "true"  # Default to True

    # Step 1: Initialize CountVectorizer
    count_vectorizer = CountVectorizer(
        inputCol="ngrams",
        outputCol="features",
        minDF=min_df,
        binary=binary
    )

    # Step 2: Fit the vectorizer model and transform the data
    vectorizer_model = count_vectorizer.fit(df)
    feature_names = vectorizer_model.vocabulary
    vocabulary_size = len(vectorizer_model.vocabulary)

    # Log vocabulary size
    logging.info(f"Vocabulary size: {vocabulary_size} (minDF={min_df}, binary={binary})")

    # Return the transformed DataFrame
    return vectorizer_model.transform(df), feature_names
