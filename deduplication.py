"""
title: deduplication service for the job listings
description: Provides a brute_force_deduplication to deduplicate an RDD with the job listings, by implementing a reduceByKey op which only keep the first text per vancancy code.
"""

import re
from typing import List, Tuple, Optional
from pyspark.rdd import RDD

def get_top_duplicates(rdd: RDD[str], n: Optional[int] = None, pattern: str = r"^\d+") -> List[Tuple[str, int]]:
    """
    Returns the top n most duplicated codes from an RDD, or all duplicates if n is None.

    Parameters:
    - rdd: The input RDD, where each entry is a job listing string.
    - n: The number of top duplicated codes to return (default is None, which returns all duplicates).
    - pattern: A regex pattern to extract the code (default is '^\d+' to match sequences of digits at the start).

    Returns:
    - A list of tuples [(code, count), ...] representing the duplicated codes and their counts.
    """

    # Step 1: Extract codes and map the RDD to (code, 1)
    keyed_rdd = rdd.map(lambda listing: (re.search(pattern, listing).group(0), 1) if re.search(pattern, listing) else None) \
                   .filter(lambda x: x is not None)  # Filter out entries where the regex did not match

    # Step 2: Reduce by key to count duplicates
    counts_rdd = keyed_rdd.reduceByKey(lambda a, b: a + b)

    # Step 3: Filter out codes with count <= 1 (no duplicates)
    duplicates_rdd = counts_rdd.filter(lambda x: x[1] > 1)

    # Step 4: Sort by count in descending order
    sorted_duplicates_rdd = duplicates_rdd.sortBy(lambda x: x[1], ascending=False)

    # If n is specified (and is a number), take the top n duplicates
    if isinstance(n, int) and n > 0:
        return sorted_duplicates_rdd.take(n)

    # Otherwise, return all duplicates
    return sorted_duplicates_rdd.collect()


def brute_force_deduplication(rdd):
    """
    Deduplicates job listings using a brute-force approach by keeping only the first occurrence
    of each job code.

    Parameters:
    - rdd: The input RDD containing job listings.

    Returns:
    - The deduplicated RDD containing unique job listings.
    """
    # Define a regex pattern to extract the sequence of uninterrupted integers at the start
    pattern = r"^\d+"

    # Step 1: Extract the code and map to (code, listing)
    keyed_rdd = rdd.map(lambda listing: (re.search(pattern, listing).group(0), listing) if re.search(pattern, listing) else None) \
                   .filter(lambda x: x is not None)  # Filter out entries where the regex did not match

    # Step 2: Deduplicate by key, keeping the first occurrence of each key
    deduplicated_rdd = keyed_rdd.reduceByKey(lambda a, b: a)  # Keep the first listing for each key

    # Step 3: Map back to the original job listings
    deduplicated_job_listings = deduplicated_rdd.map(lambda x: x[1])

    return deduplicated_job_listings
