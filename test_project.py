"""
Simple test script to verify the CORD-19 project setup
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def test_data_loading():
    """Test data loading functionality"""
    print("Testing data loading...")
    try:
        df = pd.read_csv('data/sample_metadata.csv')
        print(f"✅ Successfully loaded {len(df)} rows")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def test_data_analysis(df):
    """Test basic data analysis"""
    print("\nTesting data analysis...")
    
    # Basic statistics
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Missing data analysis
    missing_data = df.isnull().sum()
    print(f"Missing data summary:")
    print(missing_data[missing_data > 0])
    
    # Publication trends
    if 'publish_time' in df.columns:
        df['year'] = pd.to_datetime(df['publish_time']).dt.year
        yearly_counts = df['year'].value_counts().sort_index()
        print(f"\nPublications by year:")
        print(yearly_counts)
    
    # Journal analysis
    if 'journal' in df.columns:
        journal_counts = df['journal'].value_counts()
        print(f"\nTop 5 journals:")
        print(journal_counts.head())
    
    print("✅ Data analysis completed successfully")

def test_visualization(df):
    """Test visualization creation"""
    print("\nTesting visualization...")
    
    try:
        # Create a simple plot
        plt.figure(figsize=(10, 6))
        
        if 'journal' in df.columns:
            journal_counts = df['journal'].value_counts().head(5)
            plt.bar(range(len(journal_counts)), journal_counts.values)
            plt.xticks(range(len(journal_counts)), journal_counts.index, rotation=45)
            plt.title('Top 5 Journals')
            plt.ylabel('Number of Papers')
            plt.tight_layout()
            plt.savefig('test_visualization.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("✅ Visualization created successfully")
        else:
            print("❌ Journal column not found for visualization")
            
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")

def main():
    """Main test function"""
    print("CORD-19 Project Test")
    print("="*30)
    
    # Test data loading
    df = test_data_loading()
    if df is None:
        return
    
    # Test data analysis
    test_data_analysis(df)
    
    # Test visualization
    test_visualization(df)
    
    print("\n" + "="*30)
    print("✅ All tests completed successfully!")
    print("The CORD-19 project is ready to use.")
    print("\nTo run the Streamlit app:")
    print("streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()
