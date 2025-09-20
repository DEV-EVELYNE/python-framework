"""
CORD-19 Data Analysis and Visualization Script
Part 3: Data Analysis and Visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def clean_and_prepare_data(df):
    """
    Clean and prepare data for analysis
    
    Args:
        df (pd.DataFrame): Raw dataset
    
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    print("Cleaning and preparing data...")
    
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Convert publish_time to datetime
    if 'publish_time' in df_clean.columns:
        df_clean['publish_time'] = pd.to_datetime(df_clean['publish_time'], errors='coerce')
        df_clean['year'] = df_clean['publish_time'].dt.year
        df_clean['month'] = df_clean['publish_time'].dt.month
    
    # Clean text columns
    for col in ['title', 'abstract']:
        if col in df_clean.columns:
            df_clean[f'{col}_clean'] = df_clean[col].fillna('').astype(str)
            df_clean[f'{col}_word_count'] = df_clean[f'{col}_clean'].apply(lambda x: len(x.split()))
    
    # Handle missing values in key columns
    if 'journal' in df_clean.columns:
        df_clean['journal'] = df_clean['journal'].fillna('Unknown Journal')
    
    if 'source_x' in df_clean.columns:
        df_clean['source_x'] = df_clean['source_x'].fillna('Unknown Source')
    
    print(f"Data cleaning completed. Shape: {df_clean.shape}")
    return df_clean

def analyze_publication_trends(df):
    """
    Analyze publication trends over time
    
    Args:
        df (pd.DataFrame): Cleaned dataset
    
    Returns:
        dict: Analysis results
    """
    print("Analyzing publication trends...")
    
    results = {}
    
    # Publications by year
    if 'year' in df.columns:
        yearly_counts = df['year'].value_counts().sort_index()
        results['yearly_counts'] = yearly_counts
        
        # Calculate growth rate
        if len(yearly_counts) > 1:
            growth_rate = ((yearly_counts.iloc[-1] - yearly_counts.iloc[0]) / yearly_counts.iloc[0]) * 100
            results['growth_rate'] = growth_rate
    
    # Publications by month (for recent years)
    if 'month' in df.columns and 'year' in df.columns:
        recent_data = df[df['year'] >= 2020]
        if len(recent_data) > 0:
            monthly_counts = recent_data.groupby(['year', 'month']).size().reset_index(name='count')
            results['monthly_counts'] = monthly_counts
    
    return results

def analyze_journals(df):
    """
    Analyze journal publication patterns
    
    Args:
        df (pd.DataFrame): Cleaned dataset
    
    Returns:
        dict: Analysis results
    """
    print("Analyzing journal patterns...")
    
    results = {}
    
    if 'journal' in df.columns:
        # Top journals
        journal_counts = df['journal'].value_counts()
        results['top_journals'] = journal_counts.head(10)
        
        # Journal diversity
        results['total_journals'] = df['journal'].nunique()
        results['papers_per_journal'] = len(df) / df['journal'].nunique()
    
    return results

def analyze_text_content(df):
    """
    Analyze text content (titles and abstracts)
    
    Args:
        df (pd.DataFrame): Cleaned dataset
    
    Returns:
        dict: Analysis results
    """
    print("Analyzing text content...")
    
    results = {}
    
    # Title analysis
    if 'title_clean' in df.columns:
        all_titles = ' '.join(df['title_clean'].tolist())
        title_words = re.findall(r'\b\w+\b', all_titles.lower())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        filtered_words = [word for word in title_words if word not in stop_words and len(word) > 2]
        
        word_freq = Counter(filtered_words)
        results['title_word_freq'] = word_freq.most_common(20)
    
    # Abstract analysis
    if 'abstract_clean' in df.columns:
        results['avg_abstract_length'] = df['abstract_word_count'].mean()
        results['abstract_length_distribution'] = df['abstract_word_count'].describe()
    
    return results

def create_visualizations(df, analysis_results):
    """
    Create visualizations based on analysis results
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        analysis_results (dict): Results from analysis functions
    """
    print("Creating visualizations...")
    
    # Set up the plotting style
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    # 1. Publication trends over time
    if 'yearly_counts' in analysis_results:
        plt.figure(figsize=(12, 6))
        yearly_counts = analysis_results['yearly_counts']
        
        plt.subplot(1, 2, 1)
        plt.plot(yearly_counts.index, yearly_counts.values, marker='o', linewidth=2, markersize=8)
        plt.title('COVID-19 Publications Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Year')
        plt.ylabel('Number of Publications')
        plt.grid(True, alpha=0.3)
        
        # Bar chart version
        plt.subplot(1, 2, 2)
        plt.bar(yearly_counts.index, yearly_counts.values, alpha=0.7)
        plt.title('COVID-19 Publications by Year', fontsize=14, fontweight='bold')
        plt.xlabel('Year')
        plt.ylabel('Number of Publications')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('publication_trends.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 2. Top journals
    if 'top_journals' in analysis_results:
        plt.figure(figsize=(12, 8))
        top_journals = analysis_results['top_journals'].head(10)
        
        plt.barh(range(len(top_journals)), top_journals.values)
        plt.yticks(range(len(top_journals)), top_journals.index)
        plt.title('Top 10 Journals Publishing COVID-19 Research', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Publications')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, v in enumerate(top_journals.values):
            plt.text(v + 0.5, i, str(v), va='center')
        
        plt.tight_layout()
        plt.savefig('top_journals.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 3. Word cloud from titles
    if 'title_word_freq' in analysis_results:
        plt.figure(figsize=(12, 8))
        
        # Create word cloud
        word_freq_dict = dict(analysis_results['title_word_freq'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq_dict)
        
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Frequent Words in Paper Titles', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('title_wordcloud.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 4. Abstract length distribution
    if 'abstract_length_distribution' in analysis_results:
        plt.figure(figsize=(10, 6))
        
        plt.hist(df['abstract_word_count'].dropna(), bins=50, alpha=0.7, edgecolor='black')
        plt.title('Distribution of Abstract Lengths', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Words')
        plt.ylabel('Frequency')
        plt.axvline(df['abstract_word_count'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["abstract_word_count"].mean():.1f} words')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('abstract_length_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 5. Missing data visualization
    plt.figure(figsize=(12, 6))
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    plt.subplot(1, 2, 1)
    missing_data[missing_data > 0].plot(kind='bar')
    plt.title('Missing Data Count by Column', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Missing Values')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    missing_percent[missing_percent > 0].plot(kind='bar')
    plt.title('Missing Data Percentage by Column', fontsize=12, fontweight='bold')
    plt.ylabel('Percentage of Missing Values')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('missing_data_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_interactive_plots(df, analysis_results):
    """
    Create interactive plots using Plotly
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        analysis_results (dict): Analysis results
    """
    print("Creating interactive visualizations...")
    
    # 1. Interactive publication trends
    if 'yearly_counts' in analysis_results:
        fig = px.line(x=analysis_results['yearly_counts'].index, 
                     y=analysis_results['yearly_counts'].values,
                     title='COVID-19 Publications Over Time',
                     labels={'x': 'Year', 'y': 'Number of Publications'})
        fig.update_traces(mode='lines+markers', marker_size=10)
        fig.show()
    
    # 2. Interactive top journals
    if 'top_journals' in analysis_results:
        top_journals = analysis_results['top_journals'].head(15)
        fig = px.bar(x=top_journals.values, y=top_journals.index,
                    orientation='h',
                    title='Top Journals Publishing COVID-19 Research',
                    labels={'x': 'Number of Publications', 'y': 'Journal'})
        fig.update_layout(height=600)
        fig.show()
    
    # 3. Source distribution
    if 'source_x' in df.columns:
        source_counts = df['source_x'].value_counts()
        fig = px.pie(values=source_counts.values, names=source_counts.index,
                    title='Distribution of Papers by Source')
        fig.show()

def generate_summary_report(df, analysis_results):
    """
    Generate a summary report of findings
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        analysis_results (dict): Analysis results
    """
    print("\n" + "="*60)
    print("CORD-19 DATA ANALYSIS SUMMARY REPORT")
    print("="*60)
    
    print(f"\nDataset Overview:")
    print(f"- Total papers: {len(df):,}")
    print(f"- Total columns: {len(df.columns)}")
    
    if 'year' in df.columns:
        year_range = f"{df['year'].min()} - {df['year'].max()}"
        print(f"- Publication years: {year_range}")
    
    if 'total_journals' in analysis_results:
        print(f"- Number of journals: {analysis_results['total_journals']:,}")
        print(f"- Average papers per journal: {analysis_results['papers_per_journal']:.1f}")
    
    if 'growth_rate' in analysis_results:
        print(f"- Publication growth rate: {analysis_results['growth_rate']:.1f}%")
    
    if 'avg_abstract_length' in analysis_results:
        print(f"- Average abstract length: {analysis_results['avg_abstract_length']:.1f} words")
    
    print(f"\nKey Insights:")
    print(f"- The dataset contains comprehensive COVID-19 research metadata")
    print(f"- Publication trends show the rapid growth of COVID-19 research")
    print(f"- Multiple journals are contributing to COVID-19 research")
    print(f"- Text analysis reveals common themes in research titles")

def main():
    """
    Main function to run data analysis
    """
    print("CORD-19 Data Analysis and Visualization")
    print("="*50)
    
    # Load data
    try:
        df = pd.read_csv('data/metadata.csv')
        print("Loaded actual CORD-19 data")
    except FileNotFoundError:
        try:
            df = pd.read_csv('data/sample_metadata.csv')
            print("Loaded sample data")
        except FileNotFoundError:
            print("No data found. Please run data_exploration.py first.")
            return None
    
    # Clean and prepare data
    df_clean = clean_and_prepare_data(df)
    
    # Perform analysis
    analysis_results = {}
    analysis_results.update(analyze_publication_trends(df_clean))
    analysis_results.update(analyze_journals(df_clean))
    analysis_results.update(analyze_text_content(df_clean))
    
    # Create visualizations
    create_visualizations(df_clean, analysis_results)
    
    # Create interactive plots
    create_interactive_plots(df_clean, analysis_results)
    
    # Generate summary report
    generate_summary_report(df_clean, analysis_results)
    
    return df_clean, analysis_results

if __name__ == "__main__":
    df_clean, analysis_results = main()
