"""
title: Preprocessing
description: Provides tools to prepare job listings for feature discovery/extraction on standardized text.
"""

import re
import textwrap
import unicodedata

#
# (0)
#

# Define the print utility function
def print_with_wrapping(label, content, width=80):
    print(f"{label}:\n")
    print(textwrap.fill(str(content), width=width))
    print("-" * width)

# Utility functions for text preprocessing
def lowercase_text(text):
    """Lowercase the text."""
    return text.lower()

def remove_accents(text):
    """Remove accents from the string."""
    normalized = unicodedata.normalize('NFD', text)
    return ''.join(char for char in normalized if unicodedata.category(char) != 'Mn')

def remove_non_alphanumeric(text):
    """Remove non-alphanumeric characters from the string."""
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def trim_whitespaces(text):
    """Trim leading, trailing, and intermediate spaces from the string."""
    return re.sub(r'\s+', ' ', text).strip()

# Define the pipeline as individual functions
def lowercase_rdd(rdd):
    """Apply lowercase transformation to an RDD."""
    return rdd.map(lowercase_text)

def remove_accents_from_rdd(rdd):
    """Remove accents from the text in the RDD."""
    return rdd.map(remove_accents)

def remove_non_alphanumeric_from_rdd(rdd):
    """Remove non-alphanumeric characters from the text in the RDD."""
    return rdd.map(remove_non_alphanumeric)

def trim_whitespaces_rdd(rdd):
    """Trim leading, trailing, and intermediate spaces from the text in the RDD."""
    return rdd.map(trim_whitespaces)

#
# (1)
#

def standardize_text_job_listings(rdd, pipeline_steps=None, wage_pattern=r"(\d+)\s+(\d+)\s+(\d+)"):
    """
    Standardize and process raw RDD job listings by applying a series of transformations (pipeline).
    Optionally preserves wage information and returns both wages and standardized text.

    Parameters:
    - rdd: Raw RDD of job listing text lines.
    - pipeline_steps: A list of functions to apply as a pipeline (defaults to optimal pipeline).
    - wage_pattern: Regular expression to match wage triplets (code wage1 wage2 format).

    Returns:
    - RDD of tuples: (wage_info, standardized_text) for each job listing.
    """

    # Define default pipeline if none is provided
    if pipeline_steps is None:
        pipeline_steps = [
            lambda rdd: rdd.map(lambda text: text.lower()),  # Lowercase the text
            lambda rdd: rdd.map(lambda text: unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('ascii')),  # Remove accents
            lambda rdd: rdd.map(lambda text: re.sub(r'[^a-zA-Z0-9\s]', '', text)),  # Remove non-alphanumeric characters
            lambda rdd: rdd.map(lambda text: re.sub(r'\s+', ' ', text).strip())  # Trim whitespaces
        ]

    # Apply each step in the pipeline
    current_rdd = rdd
    for step in pipeline_steps:
        current_rdd = step(current_rdd)

    # Compile the wage pattern regex (looking for a code followed by two integer wages)
    wage_regex = re.compile(wage_pattern)

    def process_line(line):
        # Check if the line starts with a wage triplet (code, wage1, wage2)
        wage_match = wage_regex.match(line)
        if wage_match:
            # Extract the wage info as a triplet
            wage1 = int(wage_match.group(2))
            wage2 = int(wage_match.group(3))
            wage_info = f"{wage1}-{wage2}"  # Combine wage1 and wage2 into a single string

            # Remove the wage part from the line
            line = line[len(wage_match.group(0)):].strip()
        elif "a convenir" in line.lower():
            # Special case for "a convenir"
            wage_info = "a convenir"
            # Remove the wage part from the line
            line = re.sub(r"\s*a convenir", "", line, flags=re.IGNORECASE).strip()
        else:
            wage_info = None

        # Return a tuple of (wage_info, standardized_text)
        return (wage_info, line)

    # Return an RDD with tuples of (wage_info, standardized_text)
    return current_rdd.map(process_line)
