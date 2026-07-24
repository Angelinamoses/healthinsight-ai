import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr


def analyze_data(file):

    # Read CSV
    df = pd.read_csv(file.name)

    # Dataset Preview
    preview = df.head()

    # Dataset Info
    rows, cols = df.shape

    numeric_cols = df.select_dtypes(include="number").columns
    categorical_cols = df.select_dtypes(exclude="number").columns

    # Missing Values
    missing = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values
    })

    # Statistics
    statistics = df.describe(include="all").fillna("")

    # Histogram
    histogram = None

    if len(numeric_cols) > 0:

        plt.figure(figsize=(8,5))

        plt.hist(df[numeric_cols[0]], bins=20)

        plt.title(f"Distribution of {numeric_cols[0]}")

        plt.xlabel(numeric_cols[0])

        plt.ylabel("Frequency")

        plt.tight_layout()

        histogram = "histogram.png"

        plt.savefig(histogram)

        plt.close()

    # Correlation Heatmap
    heatmap = None

    if len(numeric_cols) > 1:

        corr = df[numeric_cols].corr()

        plt.figure(figsize=(8,6))

        plt.imshow(corr)

        plt.colorbar()

        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

        plt.yticks(range(len(corr.columns)), corr.columns)

        plt.title("Correlation Heatmap")

        plt.tight_layout()

        heatmap = "heatmap.png"

        plt.savefig(heatmap)

        plt.close()

    info = f"""
🏥 HEALTH DATASET OVERVIEW

Rows : {rows}

Columns : {cols}

Numeric Columns : {len(numeric_cols)}

Categorical Columns : {len(categorical_cols)}
"""

    return (
        preview,
        info,
        missing,
        statistics,
        histogram,
        heatmap
    )


demo = gr.Interface(

    fn=analyze_data,

    inputs=gr.File(label="📂 Upload Health Dataset (.csv)"),

    outputs=[

        gr.Dataframe(label="Dataset Preview"),

        gr.Textbox(label="Dataset Information"),

        gr.Dataframe(label="Missing Values"),

        gr.Dataframe(label="Summary Statistics"),

        gr.Image(label="Histogram"),

        gr.Image(label="Correlation Heatmap")

    ],

    title="🏥 Health Data Analyzer",

    description="""
Upload any healthcare CSV dataset.

This application performs:

• Dataset Preview

• Dataset Information

• Missing Value Analysis

• Summary Statistics

• Histogram Visualization

• Correlation Heatmap
"""

)

demo.launch(share=True)