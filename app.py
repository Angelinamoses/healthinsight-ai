"""
=========================================================
🏥 HealthInsight AI
Clinical Dataset Explorer & Quality Assessment Dashboard

Author : Angelina
Tech Stack:
- Python
- Pandas
- Matplotlib
- NumPy
- Gradio

Description:
A web application for performing exploratory data analysis
on healthcare datasets.

=========================================================
"""

# ==========================
# Import Required Libraries
# ==========================

import os
import tempfile

import gradio as gr  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import insights

# Loading Dataset

def load_dataset(file):
    """
    Load an uploaded CSV file into a Pandas DataFrame.

    Parameters
    ----------
    file : UploadedFile
        CSV file uploaded through the Gradio interface.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """

    dataframe = pd.read_csv(file.name)

    return dataframe

# Dataset Overview

def dataset_overview(dataframe):
    """
    Generate basic information about the uploaded dataset.

    Parameters
    ----------
    dataframe : pandas.DataFrame

    Returns
    -------
    dict
        Dictionary containing dataset information.
    """

    overview = {
        "Rows": dataframe.shape[0],
        "Columns": dataframe.shape[1],
        "Numeric Columns": len(
            dataframe.select_dtypes(include=np.number).columns
        ),
        "Categorical Columns": len(
            dataframe.select_dtypes(exclude=np.number).columns
        ),
        "Duplicate Records": dataframe.duplicated().sum(),
        "Missing Cells": dataframe.isnull().sum().sum(),
        "Memory Usage (KB)": round(
            dataframe.memory_usage(deep=True).sum() / 1024,
            2
        ),
    }

    return overview

# Dataset Quality Score

def calculate_quality_score(dataframe):
    """
    Calculate an overall dataset quality score.

    The score is reduced based on:
    - Missing values
    - Duplicate rows

    Returns
    -------
    tuple
        (score, status)
    """

    total_cells = dataframe.shape[0] * dataframe.shape[1]

    missing_percentage = (
        dataframe.isnull().sum().sum() / total_cells
    ) * 100

    duplicate_percentage = (
        dataframe.duplicated().sum() / len(dataframe)
    ) * 100

    score = max(
        0,
        100 - missing_percentage - duplicate_percentage
    )

    if score >= 90:
        status = "🟢 Excellent"

    elif score >= 75:
        status = "🟡 Good"

    elif score >= 50:
        status = "🟠 Fair"

    else:
        status = "🔴 Poor"

    return round(score, 2), status

# Missing value analysis

def missing_value_analysis(dataframe):
    """
    Analyze missing values in the dataset.

    Parameters
    ----------
    dataframe : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        Table containing missing values for each column.
    """

    missing_dataframe = pd.DataFrame({
        "Column": dataframe.columns,
        "Missing Values": dataframe.isnull().sum().values,
        "Missing Percentage (%)":
            (
                dataframe.isnull().sum() /
                len(dataframe)
            ).round(2) * 100
    })

    return missing_dataframe.sort_values(
        by="Missing Values",
        ascending=False
    )

# Summary statistics

def summary_statistics(dataframe):
    """
    Generate descriptive statistics for the dataset.

    Parameters
    ----------
    dataframe : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        Statistical summary.
    """

    return dataframe.describe(include="all").fillna("")

# Dataset Preview

def dataset_preview(dataframe):
    """
    Return the first five rows of the dataset.

    Parameters
    ----------
    dataframe : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """

    return dataframe.head()

# Histogram Generator

def create_histogram(dataframe):
    """
    Generate a histogram for the first numeric column.

    Parameters
    ----------
    dataframe : pandas.DataFrame

    Returns
    -------
    str
        Path to the saved histogram image.
    """

    numeric_columns = dataframe.select_dtypes(
        include=np.number
    ).columns

    if len(numeric_columns) == 0:
        return None

    column = numeric_columns[0]

    plt.figure(figsize=(8, 5))

    plt.hist(
        dataframe[column].dropna(),
        bins=20,
        edgecolor="black"
    )

    plt.title(f"Distribution of {column}")

    plt.xlabel(column)

    plt.ylabel("Frequency")

    plt.tight_layout()

    temporary_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".png"
    )

    plt.savefig(temporary_file.name)

    plt.close()

    return temporary_file.name

# Correlation heatmap

def create_heatmap(dataframe):
    """
    Generate a correlation heatmap.

    Parameters
    ----------
    dataframe : pandas.DataFrame

    Returns
    -------
    str
        Path to the saved heatmap image.
    """

    numeric_dataframe = dataframe.select_dtypes(
        include=np.number
    )

    if numeric_dataframe.shape[1] < 2:
        return None

    correlation = numeric_dataframe.corr()

    plt.figure(figsize=(8, 6))

    plt.imshow(
        correlation,
        cmap="coolwarm",
        aspect="auto"
    )

    plt.colorbar()

    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=90
    )

    plt.yticks(
        range(len(correlation.columns)),
        correlation.columns
    )

    plt.title("Feature Correlation Heatmap")

    plt.tight_layout()

    temporary_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".png"
    )

    plt.savefig(temporary_file.name)

    plt.close()

    return temporary_file.name

# Clinical Insights

def generate_clinical_insights(dataframe):
    """
    Generate human-readable insights from the uploaded dataset.

    Parameters
    ----------
    dataframe : pandas.DataFrame

    Returns
    -------
    str
        Clinical insights generated from dataset statistics.
    """

    insights = []

    # -----------------------------
    # Dataset Size
    # -----------------------------

    insights.append(
        f"• Dataset contains {len(dataframe)} patient records."
    )

    insights.append(
        f"• Total clinical features: {dataframe.shape[1]}."
    )

    # -----------------------------
    # Missing Values
    # -----------------------------

    total_missing = dataframe.isnull().sum().sum()

    if total_missing == 0:
        insights.append(
            "• No missing values detected."
        )
    else:
        insights.append(
            f"• Dataset contains {total_missing} missing values."
        )

    # -----------------------------
    # Duplicate Records
    # -----------------------------

    duplicates = dataframe.duplicated().sum()

    if duplicates == 0:
        insights.append(
            "• No duplicate patient records found."
        )
    else:
        insights.append(
            f"• Duplicate patient records: {duplicates}."
        )

    # -----------------------------
    # Numeric Feature Insights
    # -----------------------------

    numeric_columns = dataframe.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:

        mean_value = dataframe[column].mean()

        insights.append(
            f"• Average {column}: {mean_value:.2f}"
        )

    # -----------------------------
    # Strongest Correlation
    # -----------------------------

    if len(numeric_columns) > 1:

        correlation_matrix = dataframe[numeric_columns].corr().copy()

        # Remove self-correlation (diagonal)
        for i in range(len(correlation_matrix)):
            correlation_matrix.iat[i, i] = 0

        strongest = correlation_matrix.abs().stack().idxmax()

        correlation_value = correlation_matrix.loc[
            strongest[0],
            strongest[1]
        ]

        insights.append(
            f"• Strongest relationship: "
            f"{strongest[0]} ↔ {strongest[1]} "
            f"(Correlation = {correlation_value:.2f})"
        )

    return "\n".join(insights)

# Report Generator

def create_report(
    overview,
    quality_score,
    quality_status,
    insights,
):
    """
    Create a downloadable text report.

    Returns
    -------
    str
        Path to the generated report.
    """

    report = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".txt",
        mode="w",
        encoding="utf-8"
    )

    report.write(
        "=====================================\n"
    )

    report.write(
        "HealthInsight AI Report\n"
    )

    report.write(
        "=====================================\n\n"
    )

    report.write(
        "DATASET OVERVIEW\n\n"
    )

    for key, value in overview.items():

        report.write(
            f"{key}: {value}\n"
        )

    report.write("\n")

    report.write(
        f"Dataset Quality Score: {quality_score}%\n"
    )

    report.write(
        f"Quality Status: {quality_status}\n\n"
    )

    report.write(
        "CLINICAL INSIGHTS\n\n"
    )

    report.write(insights)

    report.close()

    return report.name

# Analyze Dataset Function

def analyze_dataset(file):
    """
    Main controller function.

    This function coordinates the complete
    health data analysis workflow.

    Parameters
    ----------
    file : Uploaded CSV file

    Returns
    -------
    Multiple outputs for the Gradio interface.
    """

    # -----------------------------
    # Load Dataset
    # -----------------------------

    dataframe = load_dataset(file)

    # -----------------------------
    # Dataset Preview
    # -----------------------------

    preview = dataset_preview(dataframe)

    # -----------------------------
    # Dataset Overview
    # -----------------------------

    overview = dataset_overview(dataframe)

    overview_text = ""

    for key, value in overview.items():

        overview_text += f"{key}: {value}\n"

    # -----------------------------
    # Dataset Quality
    # -----------------------------

    quality_score, quality_status = calculate_quality_score(
        dataframe
    )

    quality = (
        f"Quality Score : {quality_score}%\n"
        f"Status : {quality_status}"
    )

    # -----------------------------
    # Missing Values
    # -----------------------------

    missing = missing_value_analysis(dataframe)

    # -----------------------------
    # Statistics
    # -----------------------------

    statistics = summary_statistics(dataframe)

    # -----------------------------
    # Visualizations
    # -----------------------------

    histogram = create_histogram(dataframe)

    heatmap = create_heatmap(dataframe)

    # -----------------------------
    # AI Insights
    # -----------------------------

    insights = generate_clinical_insights(dataframe)

    # -----------------------------
    # Report
    # -----------------------------

    report = create_report(
        overview,
        quality_score,
        quality_status,
        insights
    )

    return (
        preview,
        overview_text,
        quality,
        missing,
        statistics,
        histogram,
        heatmap,
        insights,
        report
    )

# =========================================================
# Gradio User Interface
# =========================================================

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="HealthInsight AI"
) as demo:

    gr.Markdown("""
# 🏥 HealthInsight AI

### Clinical Dataset Explorer & Quality Assessment Dashboard

Upload a healthcare CSV dataset to perform exploratory data analysis,
evaluate data quality, visualize trends, and generate clinical insights.
""")

    # -------------------------
    # Input Section
    # -------------------------

    uploaded_file = gr.File(
        label="📂 Upload Clinical Dataset (.csv)"
    )

    analyze_button = gr.Button(
        "🚀 Analyze Dataset",
        variant="primary"
    )

    gr.Markdown("---")

    # -------------------------
    # Dataset Overview
    # -------------------------

    preview_output = gr.Dataframe(
        label="📋 Dataset Preview"
    )

    overview_output = gr.Textbox(
        label="📊 Dataset Overview",
        lines=8
    )

    quality_output = gr.Textbox(
        label="🏆 Data Quality Score",
        lines=3
    )

    # -------------------------
    # Analysis Tables
    # -------------------------

    missing_output = gr.Dataframe(
        label="❗ Missing Value Analysis"
    )

    statistics_output = gr.Dataframe(
        label="📈 Summary Statistics"
    )

    # -------------------------
    # Visualizations
    # -------------------------

    with gr.Row():

        histogram_output = gr.Image(
            label="📉 Histogram"
        )

        heatmap_output = gr.Image(
            label="🔥 Correlation Heatmap"
        )

    # -------------------------
    # AI Insights
    # -------------------------

    insights_output = gr.Textbox(
        label="🤖 Clinical Insights",
        lines=12
    )

    report_output = gr.File(
        label="⬇ Download Report"
    )

    # -------------------------
    # Button Action
    # -------------------------

    analyze_button.click(

        fn=analyze_dataset,

        inputs=uploaded_file,

        outputs=[
            preview_output,
            overview_output,
            quality_output,
            missing_output,
            statistics_output,
            histogram_output,
            heatmap_output,
            insights_output,
            report_output,
        ]
    )

# =========================================================
# Launch Application
# =========================================================

demo.launch(share=True)