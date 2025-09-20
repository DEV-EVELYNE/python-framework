# CORD-19 Data Analysis Project

This project analyzes the CORD-19 research dataset and presents findings through an interactive Streamlit application.

## Project Structure

```
Frameworks_Assignment/
├── requirements.txt          # Python dependencies
├── data_exploration.py       # Data loading and exploration
├── data_analysis.py          # Analysis and visualization functions
├── streamlit_app.py          # Main Streamlit application
├── README.md                 # Project documentation
└── data/                     # Data directory
    └── metadata.csv          # CORD-19 metadata file (download from Kaggle)
    └── sample_metadata.csv  # Sample data (generated automatically)
```

## Setup Instructions

1. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the CORD-19 metadata.csv file:**
   - Visit: https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge
   - Download the `metadata.csv` file
   - Place it in the `data/` directory

3. **Generate sample data:**
   ```bash
   python data_exploration.py
   ```

4. **Run the Streamlit application:**
   ```bash
   streamlit run streamlit_app.py
   ```

## Features

### 📊 Data Exploration
- Interactive data loading and exploration
- Missing data analysis
- Column information and statistics
- Sample data generation for demonstration

### 📈 Visualizations
- Publication trends over time
- Top journals analysis
- Word frequency analysis with word clouds
- Abstract length distribution
- Missing data visualization

### 🎛️ Interactive Web Application
- **Trends Tab**: Publication patterns over time
- **Journals Tab**: Journal analysis and distribution
- **Content Tab**: Text analysis and word clouds
- **Explore Tab**: Data exploration and download
- **Summary Tab**: Key insights and metrics

### 🔍 Interactive Features
- Year range filtering
- Journal selection
- Dynamic visualizations
- Data download capabilities
- Real-time metrics updates

## Learning Objectives Achieved

✅ **Practice loading and exploring real-world datasets**
- Loaded CORD-19 metadata with error handling
- Performed comprehensive data exploration
- Analyzed data structure and quality

✅ **Learn basic data cleaning techniques**
- Handled missing values appropriately
- Converted data types (dates, text)
- Created derived features (word counts, years)

✅ **Create meaningful visualizations**
- Static plots with matplotlib/seaborn
- Interactive plots with Plotly
- Word clouds for text analysis
- Comprehensive dashboard layout

✅ **Build interactive web applications**
- Modern Streamlit interface
- Multiple tabs for different analyses
- Interactive filters and controls
- Responsive design

✅ **Present data insights effectively**
- Clear visualizations
- Summary reports
- Key metrics and insights
- Professional presentation

## Key Findings

### Dataset Overview
- **Sample Dataset**: 1,000 COVID-19 research papers
- **Time Range**: 2020-2022 (for sample data)
- **Key Columns**: Title, Abstract, Authors, Journal, Publication Date

### Publication Trends
- Rapid growth in COVID-19 research publications
- Peak publication years identified
- Monthly publication patterns analyzed

### Journal Analysis
- Multiple journals contributing to COVID-19 research
- Top publishing journals identified
- Journal diversity metrics calculated

### Content Analysis
- Common themes in research titles
- Abstract length distributions
- Word frequency analysis

## Technical Implementation

### Data Processing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Datetime**: Date processing and analysis

### Visualization
- **Matplotlib/Seaborn**: Static visualizations
- **Plotly**: Interactive charts
- **WordCloud**: Text visualization

### Web Application
- **Streamlit**: Interactive web interface
- **Custom CSS**: Professional styling
- **Caching**: Performance optimization

### Data Quality
- Missing value handling
- Data type conversions
- Error handling and validation

## Usage Examples

### Running Data Exploration
```python
python data_exploration.py
```

### Running Analysis
```python
python data_analysis.py
```

### Launching Web App
```bash
streamlit run streamlit_app.py
```

## Challenges and Solutions

### Challenge 1: Large Dataset Size
**Solution**: Created sample data generator for demonstration purposes

### Challenge 2: Missing Data
**Solution**: Implemented comprehensive missing data analysis and handling strategies

### Challenge 3: Interactive Visualizations
**Solution**: Used Plotly for interactive charts and Streamlit for web interface

### Challenge 4: Performance
**Solution**: Implemented caching with `@st.cache_data` decorator

## Future Enhancements

- [ ] Add more advanced text analysis (sentiment, topics)
- [ ] Implement machine learning models for paper classification
- [ ] Add citation analysis
- [ ] Include author network analysis
- [ ] Add geographic analysis of publications
- [ ] Implement real-time data updates

## Dependencies

- pandas >= 1.3.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- streamlit >= 1.0.0
- wordcloud >= 1.8.0
- plotly >= 5.0.0
- numpy >= 1.21.0

## Dataset Source

The CORD-19 dataset is available at:
https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge

## License

This project is for educational purposes. Please refer to the original CORD-19 dataset license for data usage terms.

## Contact

For questions or issues, please refer to the project documentation or create an issue in the repository.
