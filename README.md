# BBC News Analytics Dashboard

## 🎯 Project Goal
To develop an interactive analytics platform that processes 2,200+ BBC news articles, revealing hidden linguistic patterns and semantic relationships through advanced Natural Language Processing (NLP) and dynamic visualization.

## 💾 Dataset Overview
This project analyzes the classic **BBC News Dataset**, a benchmark dataset for text classification and clustering.

*   **Total Articles**: 2,225
*   **Source**: [BBC Full Text and Category Dataset](https://www.kaggle.com/datasets/yufengdev/bbc-fulltext-and-category) (Kaggle)
*   **Data Quality**: 100% Clean (Zero missing values, Zero duplicates)
*   **Categories**: Perfectly balanced distribution across 5 domains:
    1.  **Sport** (511 articles)
    2.  **Business** (510 articles)
    3.  **Politics** (417 articles)
    4.  **Tech** (401 articles)
    5.  **Entertainment** (386 articles)

## 🛠️ Technical Skills Demonstrated
*   **Languages & Frameworks**: Python 3.8+, Dash (Flask Backend/React Frontend), Plotly Interactive
*   **Machine Learning (ML)**: Scikit-Learn, Unsupervised Learning (Hierarchical Clustering), Dimensionality Reduction (t-SNE, UMAP), Feature Engineering
*   **Natural Language Processing (NLP)**: Text Normalization, Tokenization (NLTK), TF-IDF Vectorization, N-Gram Analysis, Stopword Filtering, Vector Space Models
*   **Graph Theory & Network Science**: NetworkX, Social Network Analysis (SNA), Degree & Betweenness Centrality, Community Detection, Force-Directed Algorithms
*   **Data Engineering**: ETL Pipelines (Extract, Transform, Load), Data Cleaning, Pandas, NumPy, SciPy
*   **Advanced Visualization**: High-Dimensional Data Mapping, 3D Network Graphs, Dendrograms, Sankey Diagrams, Correlation Heatmaps, Treemaps

## 📊 Key Features
1.  **3D Knowledge Graphs**: Interactive visualization of word connections and semantic hubs using Force-Directed layouts.
2.  **Smart Document Clustering**: Automated grouping of articles using T-SNE and Hierarchical Clustering algorithms.
3.  **Cross-Category Insights**: Quantifiable correlation metrics between news domains (e.g., Business vs. Politics).
4.  **Real-Time Analytics**: Live processing pipeline that filters raw text data into insights instantly.

## 🚀 Quick Start
**1. Setup**
```bash
# Navigate to the project folder
cd BBC_News_Analytics_Dashboard
```

**2. Install Dependencies**
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

**2. Run Application**
```bash
python bbc_news_analytics_dashboard.py
```
> The dashboard will launch locally at: `http://127.0.0.1:8050/`

## 📁 Repository Structure
| File | Description |
|---|---|
| **`bbc_news_analytics_dashboard.py`** | Main Application Source Code. Contains the full NLP pipeline and server logic. |
| **`BBC_Articles_Analysis_Report.md`** | Detailed Research Findings. A comprehensive breakdown of the data insights. |
| **`visualizations/`** | Static HTML Reports. Standalone charts for offline viewing. |
| **`requirements.txt`** | Project Dependencies. |

---
## 📜 License
This project is licensed under the MIT License.

