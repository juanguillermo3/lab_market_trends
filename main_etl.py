"""
title: Main ETL
description: Transform and RDD of job postings into ML samples and stores in disk.
"""

import os
import shutil

APP_HOME = "/content/drive/MyDrive/scapping_lab_market"
os.chdir(APP_HOME)

from main_provider_curated_job_listings_rdd import provide_curated_job_listings_rdd
from preprocess import standardize_text_job_listings
from phrases import read_significant_features_from_json
from features import prepare_wages_and_features
from feature_transformation import validate_and_create_dataframe, vectorize_dataframe
from ml_samples import persist_training_partitions, read_and_transform_parquet, get_ml_samples,  get_training_metadata

#
# 0. overall application set up, and serving an unduplicated, text-standardized,
#    labelled RDD of wage, features pairs, all listings and constrained to a specific
#    segment of interest
#

# 0.1 preparing job listings and spark context
job_listings_rdd, spark = provide_curated_job_listings_rdd()  # Now returns SparkSession instead of sc
sc = spark.sparkContext  # Fetch SparkContext when needed

# 0.2 prepare significant features
significant_unigrams, significant_bigrams = read_significant_features_from_json("significant_features_export_20250127_211800.json")

# 0.3 standardadize text in job listings
standardized_job_listings = standardize_text_job_listings(job_listings_rdd)

# 0.4 filter to specific segment (eg. python related)
segment_keyword = "python"
filtered_listings = standardized_job_listings.filter(lambda line: segment_keyword.lower() in line[1].lower())  # Filter based on the text part (line[1])
filtered_listings.count()

# 0.5 extraction of significant prhases (unigrams, bigrams)
processed_sample_rdd = prepare_wages_and_features(filtered_listings, sc, significant_bigrams, significant_unigrams)
segment_size=processed_sample_rdd.count()
processed_sample_rdd.takeSample(withReplacement=True, num=3)

#
# 1.1 recast as wage ngrams dataset
#
wages_df = validate_and_create_dataframe(spark, processed_sample_rdd)
print(wages_df.count())
wages_df.show(truncate=45)

#
# 1.2 vectorize
#
vectorized_df, feature_names = vectorize_dataframe(wages_df)
vectorized_df.show()

#
# 2.0 persisting the ML samples
#

persist_training_partitions(
    vectorized_df,
    metadata_dict={
        "feature_names":feature_names,
        "full_sample_size":processed_sample_rdd.count(),
        "segment_keyword":segment_keyword}
    )
