# BBC News Analytics - Visualization Guide

## 1. Interactive Dashboard (Main App)
**Run `python dashboard.py` to access these.**
The dashboard contains the majority of the project's visualizations in a fully interactive format:

*   **Word Clouds**: Per-category most frequent terms.
*   **Knowledge Graphs**: 3D and 2D networks showing how words connect.
*   **Topic Clusters**: t-SNE and UMAP scatter plots showing semantic grouping.
*   **Correlation Heatmaps**: Interactive matrix of category similarities.
*   **Sankey Diagrams**: Flow charts showing term usage across categories.

---

## 2. Static Reference Reports (Visualizations Folder)
These 4 files are separate standalone HTML reports located in the `visualizations/` folder. They provide specific deep-dive views that are useful for quick reference without running the server.

### 📄 Article_Length_Distribution.html
*   **What it does**: Shows a histogram of how long articles are (in words and characters) for each category.
*   **Use it to**: QA the data. Check if "Sports" articles are consistently shorter than "Business" articles.

### 📄 Category_Similarity_Matrix.html
*   **What it does**: A high-detail, color-coded grid showing the exact percentage overlap between every category pair.
*   **Use it to**: See detailed correlation numbers (e.g., "Business" overlaps 48% with "Politics").

### 📄 Category_Cluster_Map.html
*   **What it does**: A simplified 2D map showing "islands" of news categories.
*   **Use it to**: Quickly visualize how distinct or similar the 5 categories are in terms of vocabulary.

### 📄 Semantic_Knowledge_Graph.html
*   **What it does**: A simplified network graph focusing purely on semantic (meaning) connections rather than just co-occurrence.
*   **Use it to**: Understand the "meaning" structure of the dataset.
