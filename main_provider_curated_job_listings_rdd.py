"""
title: single provider for the job listings RDD
description: A single source of truth for an RDD with job listings. Implements dependency injection to stack several services, such as ingestion and deduplication.
"""
# Standard library imports
import logging
from spark_session_management import create_spark_session
from ingestion_of_job_listings import construct_job_listings_rdd
from deduplication import get_top_duplicates, brute_force_deduplication  # Import deduplication functions
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Third-party imports
from dotenv import load_dotenv  # For loading configuration from environment files
load_dotenv()

# Pipeline parameters
APP_NAME = "JobListingsRDD"
APP_HOME = "/content/drive/MyDrive/scapping_lab_market"
DATE_REGEX = r"Fecha de Publicación (\d{2}/\d{2}/\d{4})"
VACANCIES_DATA_FOLDER="/content/drive/MyDrive/scapping_lab_market/scrapped"


def provide_curated_job_listings_rdd():
    # Create the SparkSession
    spark = create_spark_session()
    sc = spark.sparkContext

    # Inject dependencies to construct the RDD
    directory = VACANCIES_DATA_FOLDER  # Directory containing job listing files
    rdd, valid_files, discarded_files = construct_job_listings_rdd(sc, directory)

    if rdd:
        # Display the results of the RDD construction
        print(f"Number of valid files: {len(valid_files)}")
        #print(f"Number of discarded files: {len(discarded_files)}")
        print(f"First 5 lines of the RDD: {rdd.take(5)}")

        # Initial analysis: Total records and duplicates before deduplication
        total_records_before = rdd.count()
        all_duplicates_before = get_top_duplicates(rdd)  # Get all duplicates
        total_duplicates_before = sum([count for _, count in all_duplicates_before])
        proportion_duplicates_before = total_duplicates_before / total_records_before if total_records_before else 0

        # Print the duplicates report before deduplication
        print("\n--- Duplicates Report Before Deduplication ---")
        print(f"Total records: {total_records_before}")
        print(f"Total duplicates found: {total_duplicates_before}")
        print(f"Proportion of the sample that is duplicated: {proportion_duplicates_before:.2%}")
        print("\nTop 10 Duplicates:")
        print(f"{'Code':<10} {'Count':<10}")
        print("-" * 20)
        for code, count in all_duplicates_before[:10]:
            print(f"{code:<10} {count:<10}")

        # Brute-Force Deduplication
        deduplicated_rdd = brute_force_deduplication(rdd)

        # Post-deduplication analysis
        total_records_after = deduplicated_rdd.count()
        all_duplicates_after = get_top_duplicates(deduplicated_rdd)  # Get remaining duplicates
        total_duplicates_after = sum([count for _, count in all_duplicates_after])
        proportion_duplicates_after = total_duplicates_after / total_records_after if total_records_after else 0

        # Print the duplicates report after deduplication
        print("\n--- Duplicates Report After Deduplication ---")
        print(f"Total records: {total_records_after}")
        print(f"Total duplicates found: {total_duplicates_after}")
        print(f"Proportion of the sample that is duplicated: {proportion_duplicates_after:.2%}")
        if all_duplicates_after:
            print("\nRemaining Duplicates:")
            print(f"{'Code':<10} {'Count':<10}")
            print("-" * 20)
            for code, count in all_duplicates_after[:10]:
                print(f"{code:<10} {count:<10}")
        else:
            print("No duplicates found after deduplication.")

        # Return the deduplicated RDD and SparkContext for further processing
        return deduplicated_rdd, sc

    else:
        print("No valid RDD was created.")
        return None, sc


if __name__ == "__main__":
    job_listings, sc = provide_curated_job_listings_rdd()
