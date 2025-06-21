import pandas as pd
import plotly.express as px

def generate_eda_report(df):
    figures = []
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    for col in numeric_cols:
        fig = px.histogram(df, x=col, title=f'Univariate Analysis: {col}', template='plotly_dark')
        figures.append(fig)
    return figures

def get_statistical_summary(df):
    col = df.select_dtypes(include='number').columns[0]
    summary = f"""
Mean: {df[col].mean():.2f}
Median: {df[col].median():.2f}
Standard Deviation: {df[col].std():.2f}
Minimum: {df[col].min():.2f}
Maximum: {df[col].max():.2f}
Missing Values: {df[col].isnull().sum()}
Skewness: {df[col].skew():.2f}
Kurtosis: {df[col].kurt():.2f}
"""
    return summary
