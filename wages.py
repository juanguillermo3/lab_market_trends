"""
title: Wages Analysis
description: Plots the distribution on wages in job listings, overall vs segment specific
galleria: True
image_path: assets/wages.png
"""

# Standard library imports
import re
import pandas as pd
import matplotlib.pyplot as plt

# Local imports
# (Include any local imports here, if applicable)

# Pipeline parameters
DEFAULT_WAGE_PATTERN = r"\$[\d.]+\s*-\s*\$[\d.]+|A Convenir"
DEFAULT_SEGMENT_KEYWORD = "python"

#
# Analyze and plot wage distributions
#


# Function to calculate the midpoint of wage ranges
def calculate_midpoint(wage_range, not_specified_impute_value=10_000_000_000):
    if wage_range == "A Convenir":
        # Impute a high value to "Not Specified" to place it at the end
        return not_specified_impute_value
    else:
        # Extract integers using regex and account for commas and periods
        match = re.match(r'\$(\d+[\.,\d]*) - \$(\d+[\.,\d]*)', wage_range)
        if match:
            low, high = match.group(1), match.group(2)
            # Remove periods (thousands separator) and commas, then convert to integer
            low = int(low.replace('.', '').replace(',', ''))
            high = int(high.replace('.', '').replace(',', ''))
            return (low + high) / 2
        return None

# Updated analyze_wages function with explicit join of overall and segment distributions
def analyze_wages(rdd, wage_pattern=DEFAULT_WAGE_PATTERN, segment_keyword=DEFAULT_SEGMENT_KEYWORD, sampling_rate=None, not_specified_impute_value=10_000_000_000):
    """
    Analyzes the wage distribution of job listings, optionally filtered by a segment keyword (case-insensitive).

    Parameters:
    - rdd: The RDD of job listings.
    - wage_pattern: The regex pattern used to extract wage information. Default matches wage ranges like "$1000-$2000".
    - segment_keyword: The keyword to filter job listings (case-insensitive) for a specific segment. Default is "python".
    - sampling_rate: The optional sampling rate (between 0 and 1) to apply to the RDD. Default is None (no sampling).
    - not_specified_impute_value: The value to impute for "Not Specified" wage categories. Default is 10,000,000,000.

    Returns:
    - A plot showing the comparative wage distribution for the overall and segmented data.
    """

    # Apply sampling if provided
    if sampling_rate:
        sampled_rdd = rdd.sample(False, sampling_rate)
    else:
        sampled_rdd = rdd

    # Extract wages using the regex pattern
    def extract_wage(line):
        match = re.search(wage_pattern, line)
        return match.group(0) if match else "Not Specified"

    wages_rdd = sampled_rdd.map(lambda line: extract_wage(line))

    # Convert RDD to Pandas Series for easier manipulation
    sampled_wages = pd.Series(wages_rdd.collect())

    # Map "Not Specified" to "A Convenir"
    sampled_wages = sampled_wages.replace("A Convenir", "Not Specified")

    # Calculate relative frequency for overall wage distribution
    overall_distribution = sampled_wages.value_counts(normalize=True)

    # Calculate midwages for overall distribution
    df = pd.DataFrame(overall_distribution)
    df['midwage'] = df.index.to_series().apply(lambda x: calculate_midpoint(x, not_specified_impute_value))

    # If a segment keyword is provided, filter the RDD for those job listings
    if segment_keyword:
        segment_rdd = sampled_rdd.filter(lambda line: segment_keyword.lower() in line.lower())
        segment_wages_rdd = segment_rdd.map(lambda line: extract_wage(line))
        segment_wages = pd.Series(segment_wages_rdd.collect())
        segment_wages = segment_wages.replace("A Convenir", "Not Specified")
        segment_distribution = segment_wages.value_counts(normalize=True)
        segment_df = pd.DataFrame(segment_distribution)
        segment_df['midwage'] = segment_df.index.to_series().apply(lambda x: calculate_midpoint(x, not_specified_impute_value))
    else:
        segment_distribution = pd.Series()  # No segment, so empty

    # Sort the data by midwage (since "Not Specified" is now assigned a high value, it will be placed last)
    df = df.sort_values(by='midwage')
    if not segment_distribution.empty:
        segment_df = segment_df.sort_values(by='midwage')

    # Merge the overall and segment distributions on the index to ensure alignment
    merged_df = pd.merge(df[['proportion', 'midwage']], segment_df[['proportion', 'midwage']], left_index=True, right_index=True, how='outer', suffixes=('_overall', '_segment'))
    #return merged_df
    # Sort merged dataframe by midwage
    merged_df = merged_df.sort_values(by='midwage_overall')

    # Plotting the comparative wage distributions
    plt.figure(figsize=(12, 6))

    # Plot overall distribution
    merged_df['proportion_overall'].plot(kind='bar', color='skyblue', alpha=0.75, edgecolor='white', label="Overall")

    # Plot segment distribution if provided
    if not segment_distribution.empty:
        merged_df['proportion_segment'].plot(kind='bar', color='orange', alpha=0.75, edgecolor='white', label=f"{segment_keyword.capitalize()} Segment", position=1)

    # Customizing the plot
    plt.title(f"Comparative Wage Distribution\nOverall vs {segment_keyword.capitalize()} Segment")
    plt.xlabel("Wage Categories")
    plt.ylabel("Relative Frequency")
    plt.xticks(range(len(merged_df)), merged_df.index, rotation=90)
    plt.legend()
    plt.tight_layout()

    # Show the plot
    plt.show()

    return plt, merged_df

import re
import pandas as pd
import matplotlib.pyplot as plt

# Function to calculate the midpoint of wage ranges (private to analyze_wages)
def _calculate_midpoint(wage_range, not_specified_impute_value=10_000_000_000):
    if wage_range == "A Convenir":
        # Impute a high value to "Not Specified" to place it at the end
        return not_specified_impute_value
    else:
        # Extract integers using regex and account for commas and periods
        match = re.match(r'\$(\d+[\.,\d]*) - \$(\d+[\.,\d]*)', wage_range)
        if match:
            low, high = match.group(1), match.group(2)
            # Remove periods (thousands separator) and commas, then convert to integer
            low = int(low.replace('.', '').replace(',', ''))
            high = int(high.replace('.', '').replace(',', ''))
            return (low + high) / 2
        return None

# Function to plot nested donut charts for comparing proportions of missing wage values
def nested_donut_for_wages_missings(rdd, wage_pattern=r"\$[\d.]+\s*-\s*\$[\d.]+|A Convenir", 
                                    segment_keyword="python", sampling_rate=None, 
                                    not_specified_impute_value=10_000_000_000):
    """
    Compares the proportions of missing wage values ("Not Specified") vs provided wages,
    for both overall and segment-specific data.

    Parameters:
    - rdd: The RDD of job listings.
    - wage_pattern: The regex pattern used to extract wage information.
    - segment_keyword: The keyword to filter job listings for a specific segment.
    - sampling_rate: The optional sampling rate to apply to the RDD.
    - not_specified_impute_value: The value to impute for "Not Specified" wage categories.
    """
    
    # Apply sampling if provided
    if sampling_rate:
        sampled_rdd = rdd.sample(False, sampling_rate)
    else:
        sampled_rdd = rdd

    # Extract wages using the regex pattern
    def extract_wage(line):
        match = re.search(wage_pattern, line)
        return match.group(0) if match else "Not Specified"

    wages_rdd = sampled_rdd.map(lambda line: extract_wage(line))

    # Convert RDD to Pandas Series for easier manipulation
    sampled_wages = pd.Series(wages_rdd.collect())

    # Map "A Convenir" to "Not Specified"
    sampled_wages = sampled_wages.replace("A Convenir", "Not Specified")

    # Calculate overall proportions of "Wage Provided" vs "Not Specified"
    overall_proportions = sampled_wages.value_counts(normalize=True)
    overall_proportions = overall_proportions.get(["Not Specified", "Wage Provided"], pd.Series([0, 0]))

    # If a segment keyword is provided, filter the RDD for those job listings
    if segment_keyword:
        segment_rdd = sampled_rdd.filter(lambda line: segment_keyword.lower() in line.lower())
        segment_wages_rdd = segment_rdd.map(lambda line: extract_wage(line))
        segment_wages = pd.Series(segment_wages_rdd.collect())
        segment_wages = segment_wages.replace("A Convenir", "Not Specified")
        segment_proportions = segment_wages.value_counts(normalize=True)
        segment_proportions = segment_proportions.get(["Not Specified", "Wage Provided"], pd.Series([0, 0]))
    else:
        segment_proportions = pd.Series([0, 0])  # Empty if no segment is provided

    # Nested donut chart for "Wage Provided" vs "Not Specified"
    fig, ax = plt.subplots(figsize=(8, 8))

    # Data for donut chart
    overall_data = [overall_proportions["Wage Provided"], overall_proportions["Not Specified"]]
    segment_data = [segment_proportions["Wage Provided"], segment_proportions["Not Specified"]]

    # Create the outer donut chart (Overall)
    wedges1, texts1, autotexts1 = ax.pie(
        overall_data,
        labels=["Wage Provided", "Not Specified"],
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops=dict(width=0.4, edgecolor='white'),
        colors=['skyblue', 'lightgray']
    )

    # Create the inner donut chart (Segment)
    wedges2, texts2, autotexts2 = ax.pie(
        segment_data,
        labels=["Wage Provided", "Not Specified"],
        autopct='%1.1f%%',
        startangle=90,
        radius=0.6,
        wedgeprops=dict(width=0.4, edgecolor='white'),
        colors=['orange', 'lightyellow']
    )

    # Customizing the plot
    ax.set_title(f"Comparison of 'Wage Provided' vs 'Not Specified'\nOverall vs {segment_keyword.capitalize()} Segment")
    plt.legend(["Overall", f"{segment_keyword.capitalize()} Segment"], loc="best")

    # Show the plot
    plt.tight_layout()
    plt.show()

    return plt
