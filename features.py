"""
title: Feature Extraction
description: Perform extraction of significant features for the job listings dataset.
"""

# Standard library imports
import os
import time
import logging
import re
from datetime import datetime
from collections import defaultdict

# Third-party imports
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Local imports
from main_provider_curated_job_listings_rdd import provide_curated_job_listings_rdd
from preprocess import standardize_text_job_listings
from phrases import read_significant_features_from_json

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load parameters from .env or a similar config file
from dotenv import load_dotenv
load_dotenv()

# Pipeline parameters
APP_HOME = "/content/drive/MyDrive/scapping_lab_market"
MIN_UNIGRAM_COUNT = int(os.getenv("MIN_UNIGRAM_COUNT", 15))
LIFT_THRESHOLD = float(os.getenv("LIFT_THRESHOLD", 2.0))
REGEX_PATTERN = os.getenv("REGEX_PATTERN", r"^[a-zA-Z]+(?: [a-zA-Z]+)?$")
EXPORT_PATH = os.getenv("EXPORT_PATH", "bigrams_output.json")

# Load significant features
#significant_unigrams, significant_bigrams = read_significant_features_from_json("significant_features_export_20250123_044935.json")

#
# Feature Discovery
#

def feature_discovery_service(line, broadcasted_bigrams, broadcasted_unigrams, valid_token_expression=r"^[a-zA-Z]+$"):
    """
    Extract significant features (unigrams + bigrams) from a job listing description.

    Parameters:
    - line: The job listing text to be processed.
    - broadcasted_bigrams: Broadcasted set of significant bigrams for comparison.
    - broadcasted_unigrams: Broadcasted set of significant unigrams for comparison.
    - valid_token_expression: Regex pattern to validate tokens.

    Returns:
    - List of features: Significant unigrams and bigrams.
    """
    def generate_unigrams_and_bigrams(line, broadcasted_unigrams, valid_token_expression):
        words = line.lower().split()
        if valid_token_expression:
            words = [word for word in words if re.match(valid_token_expression, word)]

        unigrams = [word for word in words if word in broadcasted_unigrams.value]
        bigrams = [f"{unigrams[i]} {unigrams[i+1]}" for i in range(len(unigrams) - 1)]

        return unigrams, bigrams

    unigrams, bigrams = generate_unigrams_and_bigrams(line, broadcasted_unigrams, valid_token_expression)
    filtered_bigrams = [bigram for bigram in bigrams if bigram in broadcasted_bigrams.value]

    return unigrams + filtered_bigrams

#
# Wage Labeling
#

def label_wage_service(line, wage_pattern=r"(\d+)-(\d+)"):
    """
    Extract wage range from a job listing and calculate the midpoint.

    Parameters:
    - line: Job listing text containing wage information.
    - wage_pattern: Regex pattern to identify wage range.

    Returns:
    - Midpoint wage as an integer if successful; otherwise, None.
    """
    if not isinstance(line, str):
        return None

    wage_match = re.search(wage_pattern, line)
    if wage_match:
        wage_low = int(wage_match.group(1))
        wage_high = int(wage_match.group(2))
        return (wage_low + wage_high) / 2
    return None

#
# Prepare Wages and Features
#

def prepare_wages_and_features(input_rdd, sc, significant_bigrams, significant_unigrams):
    """
    Extract wages and significant features (unigrams + bigrams) from job listings.

    Parameters:
    - input_rdd: RDD of job listings, where the first entry is wage info and the second is job description.
    - sc: Spark context for broadcasting variables.
    - significant_bigrams: List of significant bigrams.
    - significant_unigrams: List of significant unigrams.

    Returns:
    - RDD of tuples containing the wage and features.
    """
    broadcasted_bigrams = sc.broadcast(significant_bigrams)
    broadcasted_unigrams = sc.broadcast(significant_unigrams)

    processed_rdd = input_rdd.map(lambda line: (
        label_wage_service(line[0]),
        feature_discovery_service(line[1], broadcasted_bigrams, broadcasted_unigrams)
    ))
    return processed_rdd

#
# Performance Estimation
#

def time_processing_on_sample(input_rdd, sample_size, target_sample_size, sc, significant_bigrams, significant_unigrams):
    """
    Estimate time to process a target number of job listings.

    Parameters:
    - input_rdd: RDD of job listings.
    - sample_size: Number of samples to process for timing.
    - target_sample_size: Target number of samples for time estimation.
    - sc: Spark context for broadcasting variables.
    - significant_bigrams: List of significant bigrams.
    - significant_unigrams: List of significant unigrams.

    Returns:
    - Estimated time to process the target number of samples.
    """
    sampled_rdd = input_rdd.takeSample(withReplacement=False, num=sample_size)
    start_time = time.time()
    prepare_wages_and_features(sc.parallelize(sampled_rdd), sc, significant_bigrams, significant_unigrams).collect()
    elapsed_time = time.time() - start_time

    return (elapsed_time / sample_size) * target_sample_size
