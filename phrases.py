"""
title: Phrases discovery
description: Provides tool for discovery of significant N-gram based phrases based on n-grams and lift metrics.
"""

# Standard library imports
import os
import re
import json
import logging
from datetime import datetime
from collections import defaultdict

# Third-party imports
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Local imports
from main_provider_curated_job_listings_rdd import provide_curated_job_listings_rdd
from preprocess import standardize_text_job_listings

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load parameters from .env or a similar config file
from dotenv import load_dotenv
load_dotenv()

# Pipeline parameters
APP_HOME = "/content/drive/MyDrive/scapping_lab_market"
MIN_UNIGRAM_COUNT = int(os.getenv("MIN_UNIGRAM_COUNT", 15))
LIFT_THRESHOLD = float(os.getenv("LIFT_THRESHOLD", 2.0))
REGEX_PATTERN = os.getenv("REGEX_PATTERN", r"^[a-zA-Z]+(?: [a-zA-Z]+)?$")  # Matches unigrams and bigrams
EXPORT_PATH = os.getenv("EXPORT_PATH", "bigrams_output.json")

# Step 1: Compute word and bigram frequencies in the corpus
def compute_word_and_bigram_frequencies(rdd):
    unigram_counts = rdd.flatMap(lambda line: line.split()).map(lambda word: (word, 1)).reduceByKey(lambda x, y: x + y)
    bigram_counts = rdd.flatMap(lambda line: [
        (f"{line.split()[i]} {line.split()[i+1]}", 1)
        for i in range(len(line.split()) - 1)
    ]).reduceByKey(lambda x, y: x + y)
    logging.debug(f"Unigram count size: {unigram_counts.count()}")
    logging.debug(f"Bigram count size: {bigram_counts.count()}")
    return unigram_counts, bigram_counts

# Step 2: Filter significant unigrams
def filter_significant_unigrams(unigram_counts, min_frequency, regex_pattern):
    pattern = re.compile(regex_pattern)
    significant_unigrams = unigram_counts.filter(
        lambda unigram: unigram[1] >= min_frequency and pattern.match(unigram[0])
    ).map(lambda unigram: unigram[0]).collect()
    logging.debug(f"Significant unigram size: {len(significant_unigrams)}")
    return significant_unigrams

# Step 3: Calculate lift scores
def calculate_lift(unigram_counts, bigram_counts, total_documents, min_unigram_count):
    unigram_frequencies = unigram_counts.collectAsMap()
    bigram_lifts = bigram_counts.map(lambda bigram: (
        bigram[0],
        -1 if unigram_frequencies.get(bigram[0].split()[0], 0) < min_unigram_count or
             unigram_frequencies.get(bigram[0].split()[1], 0) < min_unigram_count
        else bigram[1] / (
            unigram_frequencies[bigram[0].split()[0]] *
            unigram_frequencies[bigram[0].split()[1]] / total_documents
        )
    ))
    logging.debug(f"Bigram lift size: {bigram_lifts.count()}")
    return bigram_lifts

# Step 4: Filter significant bigrams
def filter_significant_bigrams(bigram_lifts, threshold, regex_pattern):
    pattern = re.compile(regex_pattern)
    filtered_bigrams = bigram_lifts.filter(
        lambda bigram: bigram[1] > threshold and pattern.match(bigram[0])
    )
    sorted_filtered_bigrams = filtered_bigrams.sortBy(lambda bigram: bigram[1], ascending=False)
    logging.debug(f"Significant bigram size: {sorted_filtered_bigrams.count()}")
    return sorted_filtered_bigrams.map(lambda bigram: bigram[0]).collect()

# Step 5: Export significant features
def export_significant_features(unigrams, bigrams, parameters, artifact_sizes, export_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"significant_features_export_{timestamp}.json"
    export_path = f"{export_dir}/{filename}"
    data = {
        "unigrams": unigrams,
        "bigrams": bigrams,
        "parameters": parameters,
        "artifact_sizes": artifact_sizes,
        "timestamp": datetime.now().isoformat()
    }
    with open(export_path, 'w') as f:
        json.dump(data, f, indent=4)
    logging.info(f"Significant features and metadata exported to {export_path}")

def read_significant_features_from_json(json_file_path):
    """
    Reads significant unigrams and bigrams from a JSON file, reports artifact sizes and parameters,
    and returns the significant features.

    Parameters:
        json_file_path (str): Path to the JSON file containing the exported significant features.

    Returns:
        tuple: A tuple containing:
            - significant_unigrams (list): List of significant unigrams.
            - significant_bigrams (list): List of significant bigrams.
    """
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        # Extract features and metadata
        significant_unigrams = data.get("unigrams", [])
        significant_bigrams = data.get("bigrams", [])
        parameters = data.get("parameters", {})
        artifact_sizes = data.get("artifact_sizes", {})

        # Report sizes and parameters
        logging.info(f"Loaded significant features from {json_file_path}")
        logging.info(f"Significant unigram size: {len(significant_unigrams)}")
        logging.info(f"Significant bigram size: {len(significant_bigrams)}")
        logging.info(f"Parameters used: {parameters}")
        logging.info(f"Artifact sizes: {artifact_sizes}")

        return significant_unigrams, significant_bigrams
    except FileNotFoundError:
        logging.error(f"File not found: {json_file_path}")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from file: {json_file_path}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise

# Main pipeline
def compute_and_export_significant_phrases():
    os.chdir(APP_HOME)
    job_listings, sc = provide_curated_job_listings_rdd()
    wages_and_text_standard_listings = standardize_text_job_listings(job_listings)
    text_standard_listings = wages_and_text_standard_listings.map(lambda tup: tup[1])
    unigram_counts, bigram_counts = compute_word_and_bigram_frequencies(text_standard_listings)
    total_documents = wages_and_text_standard_listings.count()
    bigram_lifts = calculate_lift(unigram_counts, bigram_counts, total_documents, MIN_UNIGRAM_COUNT)
    significant_unigrams = filter_significant_unigrams(unigram_counts, MIN_UNIGRAM_COUNT, REGEX_PATTERN)
    significant_bigrams = filter_significant_bigrams(bigram_lifts, LIFT_THRESHOLD, REGEX_PATTERN)
    artifact_sizes = {
        "significant_unigram_size": len(significant_unigrams),
        "significant_bigram_size": len(significant_bigrams),
    }
    export_significant_features(
        unigrams=significant_unigrams,
        bigrams=significant_bigrams,
        parameters={
            "min_unigram_count": MIN_UNIGRAM_COUNT,
            "lift_threshold": LIFT_THRESHOLD,
            "regex_pattern": REGEX_PATTERN,
            "total_documents": total_documents
        },
        artifact_sizes=artifact_sizes,
        export_dir=os.getcwd()
    )

if __name__ == "__main__":
    compute_and_export_significant_phrases()
