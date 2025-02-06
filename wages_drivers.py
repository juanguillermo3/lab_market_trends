"""
title: Wages Drivers
description: Detects the main drivers behind wage formation trough the analysis of the gradients of a Neural Network model.
galleria: True
image_path: assets/wages_drrivers.png
"""

import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Tuple

def mine_marginal_effects_with_proportion(model: tf.keras.Model, feature_names: List[str], X_data: np.ndarray) -> List[Dict[str, float]]:
    """
    Computes the marginal effects and the proportion of documents in which each feature appears.

    Parameters:
        model (tf.keras.Model): The trained model.
        feature_names (list): The list of feature names corresponding to the input features.
        X_data (numpy.ndarray): The input feature matrix (training data).

    Returns:
        sorted_marginal_effects (list): A sorted list of dictionaries containing 'keyword', 'marginal_effect', and 'prop_doc'.
    """
    # Ensure feature_names is a list of strings
    feature_names = list(map(str, feature_names))

    # Initialize storage for marginal effects and feature occurrences
    marginal_effects = {feature: [] for feature in feature_names}
    feature_occurrences = {feature: 0 for feature in feature_names}

    num_samples = X_data.shape[0]

    for i in range(num_samples):
        sample = X_data[i:i+1]  # Get a single row as input
        sample = tf.convert_to_tensor(sample, dtype=tf.float32)  # Convert to TensorFlow tensor

        with tf.GradientTape() as tape:
            tape.watch(sample)
            output = model(sample)

        gradients = tape.gradient(output, sample).numpy().flatten()  # Convert to 1D NumPy array

        # Store gradients and count occurrences
        for j, feature_name in enumerate(feature_names):
            marginal_effects[feature_name].append(gradients[j])
            if X_data[i, j] != 0:  # Count feature presence
                feature_occurrences[feature_name] += 1

    # Compute the mean marginal effect for each feature
    results = []
    for feature_name in feature_names:
        avg_marginal_effect = np.mean(marginal_effects[feature_name])  # Average across all samples
        prop_doc = feature_occurrences[feature_name] / num_samples  # Proportion of documents where feature appears

        results.append({
            "keyword": feature_name,
            "marginal_effect": avg_marginal_effect,
            "prop_doc": prop_doc
        })

    # Sort results by marginal effect in descending order
    sorted_results = sorted(results, key=lambda x: x["marginal_effect"], reverse=True)

    return sorted_results

def plot_effects_and_proportions(
    feature_importance_list: List[Dict[str, float]],
    discovery_range: Tuple[float, float] = (0.05, 0.25),
    top_n: int = 10,
    num_job_postings: str = None,  # Replaced job_postings with num_job_postings
    date_range: Tuple[pd.Timestamp, pd.Timestamp] = None
):
    """
    Create a scatter plot of relative frequency (proportion of documents) vs marginal effect for features,
    emphasizing the top N concepts with a consistent blue color scale and standardized aesthetics.

    Parameters:
        feature_importance_list (List[Dict[str, float]]): List of dictionaries, each containing:
                                                           - 'keyword': Feature name
                                                           - 'marginal_effect': Estimated marginal effect
                                                           - 'prop_doc': Proportion of documents where it appears
        discovery_range (Tuple[float, float]): Range of relative proportion to filter keywords (default: 5%-25%).
        top_n (int): Number of top concepts to emphasize (default: 10).
        num_job_postings (str): Optional parameter to indicate the number of job postings used.
        date_range (Tuple[pd.Timestamp, pd.Timestamp]): Optional date range for the data.
    """
    # Extract values from the list of dictionaries
    filtered_data = [
        feature for feature in feature_importance_list
        if discovery_range[0] <= feature["prop_doc"] <= discovery_range[1]
    ]

    if not filtered_data:
        print("No data points within the specified discovery range.")
        return

    # Create a DataFrame for plotting
    plot_data = pd.DataFrame(filtered_data)

    # Apply jittering to document proportions for better visibility
    plot_data['prop_doc'] = plot_data['prop_doc'].apply(lambda x: x + random.uniform(-0.005, 0.005))

    # Convert proportions to percentages for display
    plot_data['prop_doc'] *= 100

    # Rank the concepts by marginal effect
    plot_data['rank'] = plot_data['marginal_effect'].rank(ascending=False)
    plot_data['size'] = plot_data['rank'].apply(lambda x: 14 + (top_n-x if x <= top_n else 0))  # Emphasize top N
    plot_data['color'] = plot_data['rank'].apply(lambda x: x if x <= top_n else 0)  # Apply color to top N

    # Randomly drop labels for unranked terms (outside the top N)
    plot_data['text'] = plot_data.apply(
        lambda row: row['keyword'] if row['rank'] <= top_n or random.random() > 0.5 else '', axis=1
    )

    # Build the plot title with additional details if provided
    title = "Key Concepts with Estimated Marginal Effect"
    details = []

    if num_job_postings:
        details.append(f"over {num_job_postings} job postings")
    if date_range:
        date1 = date_range[0].strftime('%d/%m/%y')
        date2 = date_range[1].strftime('%d/%m/%y')
        details.append(f"over {date1}, {date2}")

    if details:
        title += " - " + " ".join(details)

    # Create the scatter plot with Plotly
    fig = px.scatter(
        plot_data,
        x="prop_doc",
        y="marginal_effect",
        text="text",  # Apply random label dropping logic
        color="marginal_effect",  # Color scale based on marginal effect
        title=title,
        labels={"prop_doc": "Proportion of Documents (%)", "marginal_effect": "Estimated Marginal Effect"},
        color_continuous_scale="blues",  # Standardized blue color scale
    )

    # Update scatter aesthetics
    fig.update_traces(
        marker=dict(
            size=plot_data['size'],
            opacity=0.95,  # Consistent opacity
            line=dict(color="white", width=1)  # White border for clarity
        ),
        textposition="top center",
        textfont=dict(size=10),  # Reduce font size to ~80% of default
        selector=dict(mode='markers+text')
    )

    # Add a dashed gray line at y=0
    fig.add_trace(go.Scatter(
        x=[discovery_range[0] * 100, discovery_range[1] * 100],  # Convert to percentage
        y=[0, 0],
        mode="lines",
        line=dict(color="gray", width=2, dash="dash"),  # Gray and dashed line
        showlegend=False
    ))

    # Update layout with higher frequency on x-axis ticks (every 2.5%)
    fig.update_layout(
        xaxis_title="Proportion of Documents (%)",
        yaxis_title="Estimated Marginal Effect",
        showlegend=False,  # Disable legend
        coloraxis_colorbar=dict(
            orientation="h",
            x=1,
            y=1.05  # Align legend to the top right
        ),
        margin=dict(l=100, r=20, t=50, b=40),
        xaxis=dict(
            tickmode="array",
            tickvals=np.arange(discovery_range[0] * 100, discovery_range[1] * 100 + 0.025, 2.5),  # Every 2.5 ppt
            ticktext=[f"{val:.1f}" for val in np.arange(discovery_range[0] * 100, discovery_range[1] * 100 + 0.025, 2.5)]
        ),
    )

    fig.show()

def plot_effects_rank(
    feature_importance_list: List[Dict[str, float]],
    discovery_range: Tuple[float, float] = (0.05, 0.25),
    top_n: int = 10,
    opacity: float = 0.95,  # Default opacity set to 95%
    num_job_postings: str = None,  # Replaced job_postings with num_job_postings
    date_range: Tuple[pd.Timestamp, pd.Timestamp] = None
):
    """
    Create a horizontal bar plot of the largest marginal effects for features, regardless of the proportion,
    but within the specified discovery range. The bars are blue-colored with white borders, transparent at 95%.

    Parameters:
        feature_importance_list (List[Dict[str, float]]): List of dictionaries, each containing:
                                                           - 'keyword': Feature name
                                                           - 'marginal_effect': Estimated marginal effect
                                                           - 'prop_doc': Proportion of documents where it appears
        discovery_range (Tuple[float, float]): Range of relative proportion to filter keywords (default: 5%-25%).
        top_n (int): Number of top features to display based on the largest marginal effects (default: 10).
        opacity (float): Opacity level for bars (default: 95%).
        num_job_postings (str): Optional parameter to indicate the number of job postings used.
        date_range (Tuple[pd.Timestamp, pd.Timestamp]): Optional date range for the data.
    """
    # Filter features within the discovery range
    filtered_data = [
        feature for feature in feature_importance_list
        if discovery_range[0] <= feature["prop_doc"] <= discovery_range[1]
    ]

    if not filtered_data:
        print("No data points within the specified discovery range.")
        return

    # Sort by the absolute value of marginal effect in descending order
    sorted_data = sorted(filtered_data, key=lambda x: abs(x["marginal_effect"]), reverse=True)

    # Select top_n features based on the largest marginal effects and reverse for correct ranking
    top_features = sorted_data[:top_n][::-1]  # Reverse for correct order in plotting

    # Create a DataFrame for plotting
    plot_data = pd.DataFrame(top_features)

    # Build the plot title with additional details if provided
    title = "Top Features with Largest Marginal Effects"
    details = []

    if num_job_postings:
        details.append(f"over {num_job_postings} job postings")
    if date_range:
        date1 = date_range[0].strftime('%d/%m/%y')
        date2 = date_range[1].strftime('%d/%m/%y')
        details.append(f"over {date1}, {date2}")

    if details:
        title += " - " + " ".join(details)

    # Create the horizontal bar plot with Plotly
    fig = px.bar(
        plot_data,
        y="keyword",
        x="marginal_effect",
        title=title,
        color="marginal_effect",
        color_continuous_scale="Blues",
        labels={"keyword": "Feature", "marginal_effect": "Marginal Effect"},
    )

    # Update aesthetics for the horizontal bar plot
    fig.update_traces(
        marker=dict(opacity=opacity, line=dict(color="white", width=1)),
        text=plot_data['marginal_effect'].apply(lambda x: f"{x:.2f}"),
        textposition="inside"
    )

    # Update layout with larger margins and horizontal bar chart details
    fig.update_layout(
        xaxis_title="Marginal Effect",
        yaxis_title="Feature",
        showlegend=False,  # Disable legend
        margin=dict(l=100, r=20, t=50, b=40),
    )

    fig.show()
