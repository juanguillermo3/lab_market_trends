"""
title: Dates Range
description: Describes de temporal range (dates covered) by the job listings
"""

# Standard library imports
import pandas as pd
import re
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import logging

# Third-party imports
from dotenv import load_dotenv  # For loading configuration from environment files

# Local imports
# (Include any local imports here, if applicable)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load parameters from environment configuration
load_dotenv()

# Pipeline parameters
APP_HOME = "/content/drive/MyDrive/scapping_lab_market"
DATE_REGEX = r"Fecha de Publicación (\d{2}/\d{2}/\d{4})"

#
# Dataframe with dates count from RDD of job listings
#

def extract_date_counts(job_listings, date_regex=DATE_REGEX):
    """
    Extracts and counts occurrences of dates from job listing records.

    Args:
        job_listings (RDD): RDD of job listing records as strings.
        date_regex (str): Regex pattern to extract date strings.

    Returns:
        pd.DataFrame: DataFrame with 'date' and 'count' columns.
    """
    # Map operation to extract dates and count occurrences
    date_rdd = job_listings.map(lambda record: re.search(date_regex, record)) \
                           .filter(lambda match: match is not None) \
                           .map(lambda match: (match.group(1), 1))

    # Reduce by key to count occurrences per date
    date_counts_rdd = date_rdd.reduceByKey(lambda x, y: x + y)

    # Collect as a list of dictionaries and create a DataFrame
    date_counts = date_counts_rdd.collect()
    date_counts_dicts = [{"date": k, "count": v} for k, v in date_counts]
    df = pd.DataFrame(date_counts_dicts)

    # Convert date strings to pandas datetime and sort
    df['date'] = pd.to_datetime(df['date'], format="%d/%m/%Y")
    df = df.sort_values(by='date')

    return df

#
# Plot counts of samples per date
#

def plot_listings_per_date(df, initial_date=None, end_date=None, mark_weekday=0):
    """
    Plots the number of vacancies per day, ensuring no missing dates and optionally marking specific weekdays.

    Args:
        df (pd.DataFrame): DataFrame with 'date' and 'count' columns.
        initial_date (str, optional): Start date in 'YYYY-MM-DD' format. Defaults to 2 months before today.
        end_date (str, optional): End date in 'YYYY-MM-DD' format. Defaults to today.
        mark_weekday (int, optional): Weekday to mark with dashed lines (0=Monday, 1=Tuesday, etc.). Defaults to 0 (Monday).
    """
    # Set default dates
    today = datetime.today()
    if end_date is None:
        end_date = today
    else:
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    if initial_date is None:
        initial_date = today - timedelta(days=60)
    else:
        initial_date = datetime.strptime(initial_date, "%Y-%m-%d")

    # Ensure the 'date' column is truncated to date only
    df['date'] = pd.to_datetime(df['date']).dt.date

    # Create a complete date range, truncated to date only
    full_date_range = pd.date_range(start=initial_date, end=end_date, freq='D').date

    # Reindex the DataFrame to include all dates in the range, filling missing dates with 0
    df = df.set_index('date').reindex(full_date_range, fill_value=0).reset_index()
    df.columns = ['date', 'count']  # Rename columns after reindexing

    # Convert the date column to datetime for matplotlib compatibility
    df['date'] = pd.to_datetime(df['date'])

    # Plot the data
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df['date'], df['count'], color='skyblue')

    # Set title and labels
    ax.set_title("Vacancies Per Day", fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Number of Vacancies", fontsize=12)

    # Format x-axis for daily ticks
    ax.xaxis.set_major_locator(mdates.DayLocator())  # Tick every day
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))  # Format as 'YYYY-MM-DD'

    # Highlight specific weekdays with dashed lines
    for date in df['date']:
        if date.weekday() == mark_weekday:
            ax.axvline(date, color='red', linestyle='--', linewidth=0.8, alpha=0.7, label='Marked Weekday' if date == df['date'].iloc[0] else "")

    # Rotate x-axis labels for readability
    plt.xticks(rotation=90, fontsize=10)

    # Adjust layout
    plt.tight_layout()
    #plt.show()
    return plt
