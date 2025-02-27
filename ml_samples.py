"""
title: ML samples
description: Several tools for writing/reading a vectorized dataframe as partitioned data on disk and serving the ML samples for training.
"""

import os
import numpy as np
import pyspark.sql.functions as F
from pyspark.ml.linalg import SparseVector
from pyspark.sql import SparkSession
import json
import shutil
from typing import Any, Dict
import logging
from dotenv import load_dotenv

# Set up the logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Pipeline parameters
APP_HOME = "/content/drive/MyDrive/scapping_lab_market"
TRAINING_DATA_FOLDER = os.path.join(APP_HOME, "training_data")
NUM_FOLDERS = 5

#
# (0) Persist training partitions on disk
#
def persist_training_partitions(vectorized_df: DataFrame,
                                metadata_dict: dict,
                                training_data_folder=TRAINING_DATA_FOLDER,
                                num_folders=NUM_FOLDERS):
    """
    This function persists training partitions in Parquet format, ensuring the target folder is cleared beforehand,
    and stores arbitrary metadata in a JSON file.

    Parameters:
    - vectorized_df (DataFrame): The DataFrame containing the vectorized data.
    - metadata_dict (dict): The dictionary containing arbitrary metadata to be saved as JSON.
    - training_data_folder (str): The folder where partitions will be saved.
    - num_folders (int): The number of partitions to split the data into.
    """
    # Ensure the output directory is fully cleared before writing
    if os.path.exists(training_data_folder):
        shutil.rmtree(training_data_folder)  # Deletes all existing partitions
        logger.info(f"Cleared old training data folder: {training_data_folder}")

    # Recreate the folder
    os.makedirs(training_data_folder, exist_ok=True)
    logger.info(f"Created fresh training data folder: {training_data_folder}")

    # Split the dataframe into num_folders partitions
    splits = vectorized_df.randomSplit([1.0] * num_folders, seed=42)

    # Save each partition separately in Parquet format
    for i, split_df in enumerate(splits):
        partition_path = os.path.join(training_data_folder, f"fold_{i}.parquet")
        split_df.write.mode("overwrite").parquet(partition_path)
        logger.debug(f"Saved partition {i} to {partition_path}")

    # Save metadata to a JSON file
    metadata_path = os.path.join(training_data_folder, "metadata.json")
    with open(metadata_path, 'w') as metadata_file:
        json.dump(metadata_dict, metadata_file, indent=4)
        logger.info(f"Saved metadata to {metadata_path}")

#
# (0.1) Read and transform parquet
#
def read_and_transform_parquet(spark: SparkSession, partition_path=os.path.join(TRAINING_DATA_FOLDER, "fold_0.parquet"), use_log_wage=True):
    """
    Reads a Parquet partition, extracts features and target, and converts them to NumPy arrays.
    Optionally loads log-transformed wages or original wages.

    Parameters:
        - spark (SparkSession): Spark session for reading data.
        - partition_path (str): Path to the Parquet file.
        - use_log_wage (bool): Whether to use log-transformed wages. Default is True.
    """
    logger.info(f"Reading partition from {partition_path}")

    # Load partition using SparkSession
    df = spark.read.parquet(partition_path).select("wage", "log_wage", "features")

    # Extract labels (log_wage or wage)
    y = np.array(df.select("log_wage" if use_log_wage else "wage").rdd.flatMap(lambda x: x).collect(), dtype=np.float32)

    # Convert sparse vectors to dense NumPy arrays
    def sparse_to_numpy(sparse_vec):
        dense_array = np.zeros(sparse_vec.size, dtype=np.float32)
        for idx in sparse_vec.indices:
            dense_array[idx] = 1
        return dense_array

    X = np.array(df.select("features").rdd.map(lambda row: sparse_to_numpy(row["features"])).collect())

    logger.debug(f"Processed {X.shape[0]} samples with {X.shape[1]} features")
    return X, y

#
# (1.1) Get ML samples
#
def get_ml_samples(spark: SparkSession, test_size=0.25):
    """
    Loads all partitions into memory, then applies an in-memory random partition
    into training and test sets.

    Parameters:
        - spark (SparkSession): The SparkSession for reading data.
        - test_size (float): The proportion of data to use as the test set (default 0.2).

    Returns:
        tuple: (train_X, train_y, test_X, test_y)
    """
    partition_files = [os.path.join(TRAINING_DATA_FOLDER, f)
                       for f in os.listdir(TRAINING_DATA_FOLDER) if f.endswith(".parquet")]

    all_X, all_y = [], []
    for partition_path in partition_files:
        X_batch, y_batch = read_and_transform_parquet(spark, partition_path)
        all_X.append(X_batch)
        all_y.append(y_batch)

    # Stack data into numpy arrays
    all_X = np.vstack(all_X)
    all_y = np.hstack(all_y)

    # Shuffle and split the data
    indices = np.random.permutation(len(all_X))
    split_idx = int(len(all_X) * (1 - test_size))

    train_X, test_X = all_X[indices[:split_idx]], all_X[indices[split_idx:]]
    train_y, test_y = all_y[indices[:split_idx]], all_y[indices[split_idx:]]

    logger.info(f"Total samples: {len(all_X)}")
    logger.info(f"Train samples: {len(train_X)} ({len(train_X)/len(all_X):.2%})")
    logger.info(f"Test samples: {len(test_X)} ({len(test_X)/len(all_X):.2%})")

    return train_X, train_y, test_X, test_y

#
# (1.2) Get training metadata
#
def get_training_metadata(metadata_file=os.path.join(TRAINING_DATA_FOLDER, "metadata.json")) -> Dict[str, Any]:
    """
    Reads metadata from a JSON file and returns it as a dictionary.

    Parameters:
        - metadata_file (str): Path to the metadata JSON file.

    Returns:
        dict: Metadata as a dictionary.
    """
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file {metadata_file} not found.")

    logger.info(f"Reading metadata from {metadata_file}")

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    return metadata

