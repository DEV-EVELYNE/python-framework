"""
CORD-19 Data Exploration Script
Part 1: Data Loading and Basic Exploration
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """
    Load the CORD-19 metadata CSV file
    
    Args:
        file_path (str): Path to the metadata.csv file
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        print("Loading CORD-19 metadata...")
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        print("Please download the metadata.csv file from Kaggle and place it in the data/ directory")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def basic_exploration(df):
    """
    Perform basic data exploration
    
    Args:
        df (pd.DataFrame): The dataset to explore
    """
    if df is None:
        return
    
    print("\n" + "="*50)
    print("BASIC DATA EXPLORATION")
    print("="*50)
    
    # DataFrame dimensions
    print(f"\nDataset Dimensions:")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")
    
    # Column information
    print(f"\nColumn Names and Data Types:")
    print(df.dtypes)
    
    # First few rows
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    # Basic statistics for numerical columns
    print(f"\nBasic Statistics for Numerical Columns:")
    print(df.describe())
    
    # Missing values
    print(f"\nMissing Values per Column:")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percentage': missing_percent
    })
    print(missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False))

def explore_columns(df):
    """
    Explore specific columns in detail
    
    Args:
        df (pd.DataFrame): The dataset to explore
    """
    if df is None:
        return
    
    print("\n" + "="*50)
    print("DETAILED COLUMN EXPLORATION")
    print("="*50)
    
    # Check if key columns exist
    key_columns = ['title', 'abstract', 'publish_time', 'authors', 'journal', 'source_x']
    existing_columns = [col for col in key_columns if col in df.columns]
    
    print(f"\nAvailable key columns: {existing_columns}")
    
    # Explore each key column
    for col in existing_columns:
        print(f"\n--- {col.upper()} ---")
        print(f"Data type: {df[col].dtype}")
        print(f"Non-null values: {df[col].notna().sum():,}")
        print(f"Unique values: {df[col].nunique():,}")
        
        if col in ['title', 'abstract']:
            # Show sample values for text columns
            sample_values = df[col].dropna().head(3).tolist()
            print(f"Sample values:")
            for i, val in enumerate(sample_values, 1):
                print(f"{i}. {str(val)[:100]}...")
        
        elif col == 'publish_time':
            # Show date range
            dates = pd.to_datetime(df[col], errors='coerce')
            print(f"Date range: {dates.min()} to {dates.max()}")
            print(f"Valid dates: {dates.notna().sum():,}")
        
        elif col in ['authors', 'journal', 'source_x']:
            # Show most frequent values
            top_values = df[col].value_counts().head(5)
            print(f"Top 5 values:")
            print(top_values)

def create_sample_data():
    """
    Create a sample dataset for demonstration purposes
    This is used when the actual CORD-19 data is not available
    """
    print("\nCreating sample dataset for demonstration...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sample data
    sample_data = {
        'cord_uid': [f'CORD-{i:06d}' for i in range(n_samples)],
        'title': [f'COVID-19 Research Paper {i+1}: A Study on Transmission Patterns' for i in range(n_samples)],
        'abstract': [f'This paper presents findings on COVID-19 transmission patterns in sample {i+1}. ' + 
                    'We analyzed data from multiple sources and found significant correlations.' for i in range(n_samples)],
        'publish_time': pd.date_range('2020-01-01', '2022-12-31', periods=n_samples),
        'authors': [f'Author{i%10+1}, A.; Researcher{i%5+1}, B.' for i in range(n_samples)],
        'journal': np.random.choice(['Nature', 'Science', 'Lancet', 'NEJM', 'BMJ', 'JAMA'], n_samples),
        'source_x': np.random.choice(['PubMed', 'PMC', 'arXiv', 'bioRxiv'], n_samples),
        'pdf_json_files': [f'pdf_{i}.json' for i in range(n_samples)],
        'pmc_json_files': [f'pmc_{i}.json' if i%3==0 else None for i in range(n_samples)]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Add some missing values to make it realistic
    df.loc[df.sample(frac=0.1).index, 'abstract'] = None
    df.loc[df.sample(frac=0.05).index, 'authors'] = None
    
    return df

def main():
    """
    Main function to run data exploration
    """
    print("CORD-19 Data Exploration")
    print("="*50)
    
    # Try to load actual data first
    df = load_data('data/metadata.csv')
    
    # If data not found, create sample data
    if df is None:
        print("\nActual CORD-19 data not found. Creating sample dataset...")
        df = create_sample_data()
        print("Sample dataset created successfully!")
    
    # Perform exploration
    basic_exploration(df)
    explore_columns(df)
    
    # Save sample data if we created it
    if 'CORD-' in str(df['cord_uid'].iloc[0]):
        df.to_csv('data/sample_metadata.csv', index=False)
        print(f"\nSample data saved to 'data/sample_metadata.csv'")
    
    return df

if __name__ == "__main__":
    df = main()
