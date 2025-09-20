"""
CORD-19 Data Explorer - Streamlit Application
Part 4: Interactive Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="CORD-19 Data Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff7f0e;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        df = pd.read_csv('data/metadata.csv')
        return df, "actual"
    except FileNotFoundError:
        try:
            df = pd.read_csv('data/sample_metadata.csv')
            return df, "sample"
        except FileNotFoundError:
            return None, None

@st.cache_data
def clean_data(df):
    """Clean and prepare data for analysis"""
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
    
    # Handle missing values
    if 'journal' in df_clean.columns:
        df_clean['journal'] = df_clean['journal'].fillna('Unknown Journal')
    if 'source_x' in df_clean.columns:
        df_clean['source_x'] = df_clean['source_x'].fillna('Unknown Source')
    
    return df_clean

def create_word_cloud(text_data, title):
    """Create word cloud visualization"""
    if not text_data or len(text_data) == 0:
        return None
    
    # Combine all text
    all_text = ' '.join(text_data.astype(str))
    
    # Extract words
    words = re.findall(r'\b\w+\b', all_text.lower())
    
    # Remove stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
                  'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 
                  'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 
                  'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
    
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    
    if len(filtered_words) == 0:
        return None
    
    # Create word frequency
    word_freq = Counter(filtered_words)
    
    # Create word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    
    return wordcloud

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🔬 CORD-19 Data Explorer</h1>', unsafe_allow_html=True)
    st.markdown("### Interactive exploration of COVID-19 research papers")
    
    # Load data
    df, data_type = load_data()
    
    if df is None:
        st.error("❌ No data found! Please ensure you have either 'data/metadata.csv' or 'data/sample_metadata.csv' in your data directory.")
        st.info("💡 You can run the data_exploration.py script to generate sample data.")
        return
    
    # Data type indicator
    if data_type == "sample":
        st.warning("⚠️ Using sample data for demonstration. For full analysis, download the actual CORD-19 dataset from Kaggle.")
    
    # Clean data
    df_clean = clean_data(df)
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    # Data overview
    st.sidebar.markdown("### 📊 Dataset Overview")
    st.sidebar.metric("Total Papers", f"{len(df_clean):,}")
    st.sidebar.metric("Total Columns", len(df_clean.columns))
    
    if 'year' in df_clean.columns:
        year_range = f"{df_clean['year'].min()} - {df_clean['year'].max()}"
        st.sidebar.metric("Publication Years", year_range)
    
    # Filters
    st.sidebar.markdown("### 🔍 Filters")
    
    # Year filter
    if 'year' in df_clean.columns:
        years = sorted(df_clean['year'].dropna().unique())
        year_range = st.sidebar.slider(
            "Select Year Range",
            min_value=int(min(years)),
            max_value=int(max(years)),
            value=(int(min(years)), int(max(years)))
        )
        
        # Filter data based on year range
        df_filtered = df_clean[
            (df_clean['year'] >= year_range[0]) & 
            (df_clean['year'] <= year_range[1])
        ]
    else:
        df_filtered = df_clean
    
    # Journal filter
    if 'journal' in df_clean.columns:
        journals = df_clean['journal'].value_counts().head(20).index.tolist()
        selected_journals = st.sidebar.multiselect(
            "Select Journals",
            journals,
            default=journals[:5] if len(journals) >= 5 else journals
        )
        
        if selected_journals:
            df_filtered = df_filtered[df_filtered['journal'].isin(selected_journals)]
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Trends", "📚 Journals", "📝 Content", "🔍 Explore", "📊 Summary"])
    
    with tab1:
        st.header("📈 Publication Trends")
        
        if 'year' in df_filtered.columns:
            # Yearly trends
            yearly_counts = df_filtered['year'].value_counts().sort_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Publications Over Time")
                fig = px.line(x=yearly_counts.index, y=yearly_counts.values,
                            title="COVID-19 Publications Over Time",
                            labels={'x': 'Year', 'y': 'Number of Publications'})
                fig.update_traces(mode='lines+markers', marker_size=8)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Yearly Distribution")
                fig = px.bar(x=yearly_counts.index, y=yearly_counts.values,
                           title="Publications by Year",
                           labels={'x': 'Year', 'y': 'Number of Publications'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Growth metrics
            if len(yearly_counts) > 1:
                growth_rate = ((yearly_counts.iloc[-1] - yearly_counts.iloc[0]) / yearly_counts.iloc[0]) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Peak Year", yearly_counts.idxmax(), f"{yearly_counts.max():,} papers")
                with col2:
                    st.metric("Growth Rate", f"{growth_rate:.1f}%")
                with col3:
                    st.metric("Total Years", len(yearly_counts))
        else:
            st.info("No date information available for trend analysis.")
    
    with tab2:
        st.header("📚 Journal Analysis")
        
        if 'journal' in df_filtered.columns:
            journal_counts = df_filtered['journal'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top Publishing Journals")
                top_journals = journal_counts.head(15)
                fig = px.bar(x=top_journals.values, y=top_journals.index,
                           orientation='h',
                           title="Top Journals by Publication Count",
                           labels={'x': 'Number of Publications', 'y': 'Journal'})
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Journal Distribution")
                fig = px.pie(values=journal_counts.head(10).values, 
                           names=journal_counts.head(10).index,
                           title="Top 10 Journals Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            # Journal metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Journals", f"{df_filtered['journal'].nunique():,}")
            with col2:
                st.metric("Most Prolific Journal", journal_counts.index[0], f"{journal_counts.iloc[0]:,} papers")
            with col3:
                avg_papers = len(df_filtered) / df_filtered['journal'].nunique()
                st.metric("Avg Papers per Journal", f"{avg_papers:.1f}")
        else:
            st.info("No journal information available.")
    
    with tab3:
        st.header("📝 Content Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Title Word Cloud")
            if 'title' in df_filtered.columns:
                wordcloud = create_word_cloud(df_filtered['title'], "Paper Titles")
                if wordcloud:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title('Most Frequent Words in Titles', fontsize=14, fontweight='bold')
                    st.pyplot(fig)
                else:
                    st.info("No title data available for word cloud.")
            else:
                st.info("No title information available.")
        
        with col2:
            st.subheader("Abstract Length Distribution")
            if 'abstract_word_count' in df_filtered.columns:
                fig = px.histogram(df_filtered, x='abstract_word_count',
                                 title="Distribution of Abstract Lengths",
                                 labels={'abstract_word_count': 'Number of Words', 'count': 'Frequency'})
                fig.add_vline(x=df_filtered['abstract_word_count'].mean(), 
                            line_dash="dash", line_color="red",
                            annotation_text=f"Mean: {df_filtered['abstract_word_count'].mean():.1f}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No abstract information available.")
        
        # Text statistics
        if 'title' in df_filtered.columns and 'abstract' in df_filtered.columns:
            st.subheader("📊 Text Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_title_length = df_filtered['title'].str.len().mean()
                st.metric("Avg Title Length", f"{avg_title_length:.0f} characters")
            
            with col2:
                avg_abstract_length = df_filtered['abstract_word_count'].mean()
                st.metric("Avg Abstract Length", f"{avg_abstract_length:.0f} words")
            
            with col3:
                total_titles = df_filtered['title'].notna().sum()
                st.metric("Papers with Titles", f"{total_titles:,}")
            
            with col4:
                total_abstracts = df_filtered['abstract'].notna().sum()
                st.metric("Papers with Abstracts", f"{total_abstracts:,}")
    
    with tab4:
        st.header("🔍 Data Explorer")
        
        # Data sample
        st.subheader("📋 Sample Data")
        
        # Show sample of filtered data
        sample_size = st.slider("Number of rows to display", 5, 50, 10)
        st.dataframe(df_filtered.head(sample_size))
        
        # Download option
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name=f"cord19_filtered_data_{len(df_filtered)}_rows.csv",
            mime="text/csv"
        )
        
        # Column information
        st.subheader("📊 Column Information")
        col_info = pd.DataFrame({
            'Column': df_filtered.columns,
            'Data Type': df_filtered.dtypes,
            'Non-Null Count': df_filtered.notna().sum(),
            'Null Count': df_filtered.isnull().sum(),
            'Unique Values': df_filtered.nunique()
        })
        st.dataframe(col_info)
        
        # Missing data visualization
        st.subheader("🔍 Missing Data Analysis")
        missing_data = df_filtered.isnull().sum()
        missing_percent = (missing_data / len(df_filtered)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing Percentage': missing_percent.values
        }).sort_values('Missing Count', ascending=False)
        
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        
        if len(missing_df) > 0:
            fig = px.bar(missing_df, x='Missing Percentage', y='Column',
                        orientation='h',
                        title="Missing Data by Column",
                        labels={'Missing Percentage': 'Percentage Missing', 'Column': 'Column Name'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing data in the filtered dataset!")
    
    with tab5:
        st.header("📊 Summary Report")
        
        # Key metrics
        st.subheader("🎯 Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Papers", f"{len(df_filtered):,}")
        
        with col2:
            if 'journal' in df_filtered.columns:
                st.metric("Unique Journals", f"{df_filtered['journal'].nunique():,}")
            else:
                st.metric("Unique Journals", "N/A")
        
        with col3:
            if 'year' in df_filtered.columns:
                year_range = f"{df_filtered['year'].min()}-{df_filtered['year'].max()}"
                st.metric("Year Range", year_range)
            else:
                st.metric("Year Range", "N/A")
        
        with col4:
            if 'source_x' in df_filtered.columns:
                st.metric("Data Sources", f"{df_filtered['source_x'].nunique():,}")
            else:
                st.metric("Data Sources", "N/A")
        
        # Insights
        st.subheader("💡 Key Insights")
        
        insights = []
        
        if 'year' in df_filtered.columns and len(df_filtered['year'].dropna()) > 0:
            peak_year = df_filtered['year'].value_counts().idxmax()
            peak_count = df_filtered['year'].value_counts().max()
            insights.append(f"📈 Peak publication year was {peak_year} with {peak_count:,} papers")
        
        if 'journal' in df_filtered.columns:
            top_journal = df_filtered['journal'].value_counts().index[0]
            top_count = df_filtered['journal'].value_counts().iloc[0]
            insights.append(f"📚 Most prolific journal is '{top_journal}' with {top_count:,} papers")
        
        if 'abstract_word_count' in df_filtered.columns:
            avg_length = df_filtered['abstract_word_count'].mean()
            insights.append(f"📝 Average abstract length is {avg_length:.0f} words")
        
        if len(df_filtered) > 0:
            insights.append(f"🔍 Current filters show {len(df_filtered):,} papers out of {len(df_clean):,} total")
        
        for insight in insights:
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
        
        # Data quality assessment
        st.subheader("🔍 Data Quality Assessment")
        
        quality_metrics = []
        
        if 'title' in df_filtered.columns:
            title_coverage = (df_filtered['title'].notna().sum() / len(df_filtered)) * 100
            quality_metrics.append(f"📄 Title coverage: {title_coverage:.1f}%")
        
        if 'abstract' in df_filtered.columns:
            abstract_coverage = (df_filtered['abstract'].notna().sum() / len(df_filtered)) * 100
            quality_metrics.append(f"📝 Abstract coverage: {abstract_coverage:.1f}%")
        
        if 'authors' in df_filtered.columns:
            author_coverage = (df_filtered['authors'].notna().sum() / len(df_filtered)) * 100
            quality_metrics.append(f"👥 Author coverage: {author_coverage:.1f}%")
        
        for metric in quality_metrics:
            st.info(metric)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🔬 CORD-19 Data Explorer | Built with Streamlit | 
        <a href='https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge' target='_blank'>Dataset Source</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
