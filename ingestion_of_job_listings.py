
"""
title: ingestions service for the job listings
description: Reads job listings from a disk folder where scrapped data is located. Discard files that dont follow the expected structure.
"""

import os
import gzip
import re
import logging

# Regular expression to validate date patterns in files
DEFAULT_DATE_PATTERN = re.compile(r'\d{2}/\d{2}/\d{4}$')
DEFAULT_FILE_PATTERN = r'^vacancies_.*\.txt\.gz$'
VACANCIES_DATA_FOLDER = "/content/drive/MyDrive/scapping_lab_market"

def construct_job_listings_rdd(spark_context, directory=VACANCIES_DATA_FOLDER, file_pattern=DEFAULT_FILE_PATTERN, date_pattern=DEFAULT_DATE_PATTERN):
    """
    Constructs an RDD from valid job listing files in the given directory.

    Args:
        spark_context: The SparkContext to use for RDD creation.
        directory: The directory to search for files. Defaults to VACANCIES_DATA_FOLDER.
        file_pattern: The regular expression pattern to match file names. Defaults to matching 'vacancies_' prefixed .txt.gz files.
        date_pattern: The regular expression pattern to validate file content (default is for date format).

    Returns:
        A tuple of (RDD, list_of_valid_files, list_of_discarded_files).
    """
    valid_files = []
    discarded_files = []

    for filename in os.listdir(directory):
        # Apply the file name pattern to filter relevant files
        if re.match(file_pattern, filename):
            file_path = os.path.join(directory, filename)

            try:
                # Open the gzip file and read the first 10 lines
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    lines = [next(f).strip() for _ in range(10)]
            except Exception as e:
                # Log any errors encountered while reading the file
                logging.warning(f"Error reading file {filename}: {e}")
                discarded_files.append(file_path)
                continue

            # Check if all lines match the provided content pattern
            if all(date_pattern.search(line) for line in lines):
                valid_files.append(file_path)
                logging.info(f"File {filename} passed validation and will be included.")
            else:
                logging.warning(f"File {filename} skipped: Not all lines match the expected format.")
                discarded_files.append(file_path)

    if valid_files:
        rdd = spark_context.textFile(",".join(valid_files))
        logging.info(f"RDD created successfully with {len(valid_files)} valid files.")
        return rdd, valid_files, discarded_files
    else:
        logging.warning("No valid files found. RDD not created.")
        return None, valid_files, discarded_files
