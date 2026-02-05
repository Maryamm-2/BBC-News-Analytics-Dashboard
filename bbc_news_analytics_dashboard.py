"""
BBC News Dataset - Advanced Interactive Analytics Dashboard (FIXED & TESTED)
Complete Text Mining & Visualization Suite

INSTALLATION GUIDE:
==================
Python Version: 3.9 (Recommended)

Step-by-step Installation:
---------------------------
1. Create virtual environment:
   python -m venv bbc_env

2. Activate environment:
   Windows: bbc_env\Scripts\activate
   Mac/Linux: source bbc_env/bin/activate

3. Install packages (EXACT versions - tested):
   pip install pandas==1.5.3
   pip install numpy==1.24.3
   pip install matplotlib==3.7.1
   pip install plotly==5.14.1
   pip install dash==2.9.3
   pip install dash-bootstrap-components==1.4.1
   pip install networkx==3.1
   pip install scikit-learn==1.2.2
   pip install scipy==1.10.1


5. Run once for NLTK data:
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

Author: Advanced Analytics Team
Version: 4.0 - Fixed & Production Ready
"""

import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter, defaultdict
import re
import warnings
import json

warnings.filterwarnings("ignore")

# Visualization libraries
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Dash framework
import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc

# Text processing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage

# Optional UMAP - DISABLED (using t-SNE instead for Windows compatibility)
UMAP_AVAILABLE = False

# Optional NLTK
try:
    import nltk
    from nltk.corpus import stopwords

    nltk.data.path.append("./nltk_data")
    try:
        NLTK_STOPWORDS = set(stopwords.words("english"))
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download("stopwords", quiet=True)
        NLTK_STOPWORDS = set(stopwords.words("english"))
except ImportError:
    NLTK_STOPWORDS = set()
    print(" [WARN]  NLTK not available. Using custom stopwords.")

# ============================================================================
# ADVANCED TEXT PREPROCESSOR
# ============================================================================


class AdvancedTextPreprocessor:
    """Advanced text preprocessing with comprehensive stopword removal"""

    def __init__(self):
        # Comprehensive stopwords
        self.stopwords = set(
            [
                # Common English stopwords
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "from",
                "about",
                "as",
                "into",
                "through",
                "during",
                "before",
                "after",
                "above",
                "below",
                "up",
                "down",
                "out",
                "off",
                "over",
                "under",
                "again",
                "further",
                "then",
                "once",
                "here",
                "there",
                "when",
                "where",
                "why",
                "how",
                "all",
                "both",
                "each",
                "few",
                "more",
                "most",
                "other",
                "some",
                "such",
                "no",
                "nor",
                "not",
                "only",
                "own",
                "same",
                "so",
                "than",
                "too",
                "very",
                "s",
                "t",
                "can",
                "will",
                "just",
                "don",
                "should",
                "now",
                "does",
                "did",
                "doing",
                # Pronouns
                "i",
                "me",
                "my",
                "myself",
                "we",
                "our",
                "ours",
                "ourselves",
                "you",
                "your",
                "yours",
                "yourself",
                "yourselves",
                "he",
                "him",
                "his",
                "himself",
                "she",
                "her",
                "hers",
                "herself",
                "it",
                "its",
                "itself",
                "they",
                "them",
                "their",
                "theirs",
                "themselves",
                "what",
                "which",
                "who",
                "whom",
                "this",
                "that",
                "these",
                "those",
                # Verbs
                "am",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "being",
                "have",
                "has",
                "had",
                "having",
                "do",
                "does",
                "did",
                "doing",
                "would",
                "could",
                "should",
                "might",
                "must",
                "shall",
                "may",
                "can",
                "will",
                # Common in news
                "said",
                "would",
                "also",
                "people",
                "one",
                "two",
                "three",
                "four",
                "five",
                "new",
                "first",
                "last",
                "year",
                "years",
                "much",
                "many",
                "mr",
                "ms",
                "mrs",
                "told",
                "says",
                "according",
                "get",
                "got",
                "make",
                "made",
                "take",
                "taken",
                "go",
                "going",
                "went",
                "come",
                "came",
                "well",
                "even",
                "say",
                "like",
                "time",
                "back",
                "use",
                "used",
                "around",
                "another",
                "still",
                "since",
                "way",
                "us",
                "uk",
                "bbc",
                "pm",
                "am",
                "per",
                "via",
                "could",
                "day",
                "week",
                "month",
                "monday",
                "tuesday",
                "wednesday",
                "thursday",
                "friday",
                "saturday",
                "sunday",
            ]
        )

        # Add NLTK stopwords if available
        if NLTK_STOPWORDS:
            self.stopwords.update(NLTK_STOPWORDS)

    def clean_text(self, text):
        """Deep cleaning of text"""
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"\S+@\S+", "", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        text = " ".join(text.split())

        return text

    def tokenize(self, text):
        """Tokenize and filter"""
        text = self.clean_text(text)
        tokens = text.split()

        tokens = [
            t
            for t in tokens
            if t not in self.stopwords and len(t) > 3 and not t.isdigit()
        ]

        return tokens

    def get_clean_text(self, text):
        """Get cleaned text as string"""
        tokens = self.tokenize(text)
        return " ".join(tokens)


# ============================================================================
# BBC NEWS ANALYZER
# ============================================================================


class BBCNewsAnalyzerAdvanced:
    """Advanced text analytics"""

    def __init__(self, csv_path=None):
        print("[INFO] Loading dataset...")

        # If no path provided, try to get from kagglehub
        if csv_path is None:
            try:
                import kagglehub
                import os

                dataset_path = kagglehub.dataset_download(
                    "yufengdev/bbc-fulltext-and-category"
                )
                csv_path = os.path.join(dataset_path, "bbc-text.csv")
            except Exception:
                csv_path = "bbc-text.csv"

        self.df = pd.read_csv(csv_path)
        self.preprocessor = AdvancedTextPreprocessor()
        self.categories = sorted(self.df["category"].unique())

        print("[STEP] Preprocessing text...")
        self.df["clean_text"] = self.df["text"].apply(self.preprocessor.get_clean_text)
        self.df = self.df[self.df["clean_text"].str.len() > 50]

        print(
            f"[SUCCESS] Loaded {len(self.df)} documents across {len(self.categories)} categories"
        )

        self.results = {}

    def calculate_tfidf(self, max_features=100):
        """Calculate TF-IDF"""
        print("[STEP] Calculating TF-IDF scores...")

        vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=(1, 2), min_df=2, max_df=0.8
        )

        tfidf_matrix = vectorizer.fit_transform(self.df["clean_text"])
        feature_names = vectorizer.get_feature_names_out()

        category_tfidf = {}
        for category in self.categories:
            cat_texts = self.df[self.df["category"] == category]["clean_text"]
            cat_tfidf = vectorizer.transform(cat_texts)
            avg_tfidf = np.asarray(cat_tfidf.mean(axis=0)).flatten()
            category_tfidf[category] = dict(zip(feature_names, avg_tfidf))

        self.results["tfidf_matrix"] = tfidf_matrix
        self.results["tfidf_features"] = feature_names
        self.results["category_tfidf"] = category_tfidf

        return category_tfidf

    def extract_term_frequencies(self, top_n=50):
        """Extract term frequencies"""
        print("[STEP] Calculating term frequencies...")

        term_freq = {}
        for category in self.categories:
            cat_texts = " ".join(self.df[self.df["category"] == category]["clean_text"])
            tokens = cat_texts.split()
            term_freq[category] = Counter(tokens).most_common(top_n)

        self.results["term_freq"] = term_freq
        return term_freq

    def calculate_cooccurrence(self, window_size=5, top_n=100):
        """Calculate co-occurrence"""
        print("[STEP] Extracting co-occurrence patterns...")

        cooccurrence = defaultdict(int)

        for text in self.df["clean_text"]:
            tokens = text.split()

            for i in range(len(tokens)):
                for j in range(i + 1, min(i + window_size, len(tokens))):
                    pair = tuple(sorted([tokens[i], tokens[j]]))
                    cooccurrence[pair] += 1

        cooccurrence_sorted = sorted(
            cooccurrence.items(), key=lambda x: x[1], reverse=True
        )[:top_n]

        self.results["cooccurrence"] = cooccurrence_sorted
        return cooccurrence_sorted

    def build_knowledge_graphs(self, terms_per_category=10):
        """Build knowledge graphs"""
        print("[STEP] Building knowledge graphs...")

        # Graph 1: Category-Term Network
        G_main = nx.Graph()

        if "category_tfidf" not in self.results:
            self.calculate_tfidf()

        for category in self.categories:
            G_main.add_node(category, node_type="category", size=60, color="#FF6B6B")

            top_terms = sorted(
                self.results["category_tfidf"][category].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:terms_per_category]

            for term, score in top_terms:
                if not G_main.has_node(term):
                    G_main.add_node(term, node_type="term", size=20, color="#4ECDC4")
                G_main.add_edge(category, term, weight=score * 100)

        # Graph 2: Co-occurrence Network
        G_cooc = nx.Graph()

        if "cooccurrence" not in self.results:
            self.calculate_cooccurrence()

        for (term1, term2), count in self.results["cooccurrence"][:50]:
            G_cooc.add_edge(term1, term2, weight=count)

        self.results["knowledge_graphs"] = {"main": G_main, "cooccurrence": G_cooc}

        return G_main, G_cooc

    def calculate_tsne_embeddings(self):
        """Calculate t-SNE"""
        print("[STEP] Calculating t-SNE embeddings...")

        if "tfidf_matrix" not in self.results:
            self.calculate_tfidf(max_features=200)

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings = tsne.fit_transform(self.results["tfidf_matrix"].toarray())

        self.results["tsne_embeddings"] = embeddings
        return embeddings

    def calculate_umap_embeddings(self):
        """Calculate UMAP embeddings - now uses t-SNE for compatibility"""
        print("[STEP] Calculating t-SNE embeddings (UMAP alternative)...")
        return self.calculate_tsne_embeddings()

    def build_hierarchical_clustering(self):
        """Build hierarchical clustering"""
        print("[STEP] Building hierarchical clustering...")

        if "category_tfidf" not in self.results:
            self.calculate_tfidf()

        vectors = []
        for category in self.categories:
            vec = [
                self.results["category_tfidf"][category].get(f, 0)
                for f in self.results["tfidf_features"]
            ]
            vectors.append(vec)

        linkage_matrix = linkage(vectors, method="ward")

        self.results["linkage_matrix"] = linkage_matrix
        return linkage_matrix

    def analyze_all(self):
        """Run all analyses"""
        print("\n" + "=" * 70)
        print("STARTING COMPREHENSIVE ANALYSIS")
        print("=" * 70 + "\n")

        self.calculate_tfidf()
        self.extract_term_frequencies()
        self.calculate_cooccurrence()
        self.build_knowledge_graphs()
        self.calculate_tsne_embeddings()

        # Store t-SNE as UMAP for compatibility
        self.results["umap_embeddings"] = self.results["tsne_embeddings"]

        self.build_hierarchical_clustering()

        print("\n[SUCCESS] All analyses complete!\n")
        return self.results


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================


def create_wordcloud_plotly(term_freq, category, title):
    """Interactive word cloud"""
    words = [term for term, _ in term_freq[:30]]
    frequencies = [freq for _, freq in term_freq[:30]]

    if not words:
        return go.Figure().add_annotation(text="No data available", showarrow=False)

    max_freq = max(frequencies)
    sizes = [20 + (freq / max_freq) * 60 for freq in frequencies]

    n = len(words)
    angles = np.linspace(0, 2 * np.pi, n)
    radii = np.random.uniform(0.3, 1, n)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)

    colors = px.colors.sample_colorscale(
        "Viridis", [freq / max_freq for freq in frequencies]
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="text",
            text=words,
            textfont=dict(size=sizes, color=colors),
            hovertemplate="<b>%{text}</b><br>Frequency: %{customdata}<extra></extra>",
            customdata=frequencies,
        )
    )

    fig.update_layout(
        title=title,
        showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        height=600,
        plot_bgcolor="white",
    )

    return fig


def create_tag_cloud_bubbles(term_freq, category, title):
    """Bubble tag cloud"""
    words = [term for term, _ in term_freq[:30]]
    frequencies = [freq for _, freq in term_freq[:30]]

    if not words:
        return go.Figure().add_annotation(text="No data available", showarrow=False)

    n = len(words)
    x = np.random.uniform(0, 10, n)
    y = np.random.uniform(0, 10, n)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers+text",
            marker=dict(
                size=[f / max(frequencies) * 100 for f in frequencies],
                color=frequencies,
                colorscale="Plasma",
                showscale=True,
                colorbar=dict(title="Frequency"),
                line=dict(width=2, color="white"),
            ),
            text=words,
            textposition="middle center",
            textfont=dict(size=10, color="white"),
            hovertemplate="<b>%{text}</b><br>Frequency: %{marker.color}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        showlegend=False,
        xaxis=dict(
            showgrid=False, showticklabels=False, zeroline=False, range=[-1, 11]
        ),
        yaxis=dict(
            showgrid=False, showticklabels=False, zeroline=False, range=[-1, 11]
        ),
        height=600,
        plot_bgcolor="#f8f9fa",
    )

    return fig


def create_network_graph_3d(G, title):
    """3D network visualization"""
    pos = nx.spring_layout(G, dim=3, seed=42, k=2)

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_z = [pos[node][2] for node in G.nodes()]

    node_text = list(G.nodes())
    node_sizes = [G.nodes[node].get("size", 20) for node in G.nodes()]
    node_colors = [G.nodes[node].get("color", "#4ECDC4") for node in G.nodes()]

    edge_x, edge_y, edge_z = [], [], []

    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line=dict(color="#888", width=1),
        hoverinfo="none",
        showlegend=False,
    )

    node_trace = go.Scatter3d(
        x=node_x,
        y=node_y,
        z=node_z,
        mode="markers+text",
        marker=dict(
            size=node_sizes, color=node_colors, line=dict(color="white", width=1)
        ),
        text=node_text,
        textposition="top center",
        textfont=dict(size=10),
        hovertemplate="<b>%{text}</b><extra></extra>",
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, showticklabels=False, title=""),
            zaxis=dict(showgrid=False, showticklabels=False, title=""),
            bgcolor="rgba(240,240,240,0.9)",
        ),
        height=800,
        showlegend=False,
    )

    return fig


def create_force_directed_graph(analyzer):
    """Force-directed graph"""
    G = analyzer.results["knowledge_graphs"]["main"]
    pos = nx.kamada_kawai_layout(G)

    categories = [n for n in G.nodes() if G.nodes[n].get("node_type") == "category"]
    terms = [n for n in G.nodes() if G.nodes[n].get("node_type") == "term"]

    fig = go.Figure()

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = G[edge[0]][edge[1]].get("weight", 1)

        fig.add_trace(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(color="rgba(150,150,150,0.2)", width=weight / 20),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    cat_x = [pos[n][0] for n in categories]
    cat_y = [pos[n][1] for n in categories]

    fig.add_trace(
        go.Scatter(
            x=cat_x,
            y=cat_y,
            mode="markers+text",
            marker=dict(size=50, color="#FF6B6B", line=dict(width=3, color="#C92A2A")),
            text=categories,
            textposition="middle center",
            textfont=dict(size=11, color="white"),
            name="Categories",
            hovertemplate="<b>%{text}</b><extra></extra>",
        )
    )

    term_x = [pos[n][0] for n in terms]
    term_y = [pos[n][1] for n in terms]

    fig.add_trace(
        go.Scatter(
            x=term_x,
            y=term_y,
            mode="markers+text",
            marker=dict(size=20, color="#4ECDC4", line=dict(width=2, color="#0B7285")),
            text=terms,
            textposition="top center",
            textfont=dict(size=9),
            name="Terms",
            hovertemplate="<b>%{text}</b><extra></extra>",
        )
    )

    fig.update_layout(
        title="Force-Directed Knowledge Graph",
        showlegend=True,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        height=900,
        plot_bgcolor="#f8f9fa",
    )

    return fig


def create_sankey_category_flow(analyzer):
    """Sankey diagram"""
    categories = analyzer.categories
    source_indices = []
    target_indices = []
    values = []

    all_nodes = list(categories)
    node_colors = ["#FF6B6B"] * len(categories)

    for cat_idx, category in enumerate(categories):
        top_terms = sorted(
            analyzer.results["category_tfidf"][category].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        for term, score in top_terms:
            if term not in all_nodes:
                all_nodes.append(term)
                node_colors.append("#4ECDC4")

            source_indices.append(cat_idx)
            target_indices.append(all_nodes.index(term))
            values.append(score * 1000)

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=all_nodes,
                    color=node_colors,
                ),
                link=dict(
                    source=source_indices,
                    target=target_indices,
                    value=values,
                    color="rgba(100,150,200,0.3)",
                ),
            )
        ]
    )

    fig.update_layout(
        title="Term Flow: Categories to Important Terms", height=700, font=dict(size=10)
    )

    return fig


def create_tsne_scatter(analyzer):
    """t-SNE scatter plot"""
    if "tsne_embeddings" not in analyzer.results:
        analyzer.calculate_tsne_embeddings()

    embeddings = analyzer.results["tsne_embeddings"]

    df_plot = pd.DataFrame(
        {
            "x": embeddings[:, 0],
            "y": embeddings[:, 1],
            "category": analyzer.df["category"].values,
            "text_preview": analyzer.df["text"].str[:100] + "...",
        }
    )

    fig = px.scatter(
        df_plot,
        x="x",
        y="y",
        color="category",
        hover_data=["text_preview"],
        title="t-SNE Document Clustering",
        color_discrete_sequence=px.colors.qualitative.Set2,
        height=700,
    )

    fig.update_traces(
        marker=dict(size=8, opacity=0.7, line=dict(width=1, color="white"))
    )

    return fig


def create_umap_scatter(analyzer):
    """UMAP scatter plot - now uses t-SNE"""
    if "umap_embeddings" not in analyzer.results:
        analyzer.results["umap_embeddings"] = analyzer.results.get("tsne_embeddings")

        if "tsne_embeddings" not in analyzer.results:
            analyzer.calculate_tsne_embeddings()
            analyzer.results["umap_embeddings"] = analyzer.results["tsne_embeddings"]

    embeddings = analyzer.results["umap_embeddings"]

    df_plot = pd.DataFrame(
        {
            "x": embeddings[:, 0],
            "y": embeddings[:, 1],
            "category": analyzer.df["category"].values,
            "text_preview": analyzer.df["text"].str[:100] + "...",
        }
    )

    fig = px.scatter(
        df_plot,
        x="x",
        y="y",
        color="category",
        hover_data=["text_preview"],
        title="Alternative t-SNE Clustering (High Quality)",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        height=700,
    )

    fig.update_traces(marker=dict(size=8, opacity=0.7))

    return fig


def create_dendrogram_plot(analyzer):
    """Hierarchical clustering"""
    if "linkage_matrix" not in analyzer.results:
        analyzer.build_hierarchical_clustering()

    linkage_matrix = analyzer.results["linkage_matrix"]
    dend = dendrogram(linkage_matrix, labels=analyzer.categories, no_plot=True)

    icoord = np.array(dend["icoord"])
    dcoord = np.array(dend["dcoord"])

    fig = go.Figure()

    for i in range(len(icoord)):
        fig.add_trace(
            go.Scatter(
                x=icoord[i],
                y=dcoord[i],
                mode="lines",
                line=dict(color="#2C3E50", width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    labels = dend["ivl"]
    x_labels = [5, 15, 25, 35, 45][: len(labels)]

    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=[0] * len(labels),
            mode="text",
            text=labels,
            textposition="bottom center",
            textfont=dict(size=12),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title="Hierarchical Clustering Dendrogram",
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=True, title="Distance"),
        height=600,
        plot_bgcolor="white",
    )

    return fig


def create_sunburst_hierarchy(analyzer):
    """Sunburst chart"""
    data = []

    for category in analyzer.categories:
        data.append(
            {
                "labels": category,
                "parents": "",
                "values": len(analyzer.df[analyzer.df["category"] == category]),
            }
        )

        top_terms = sorted(
            analyzer.results["category_tfidf"][category].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:8]

        for term, score in top_terms:
            data.append({"labels": term, "parents": category, "values": score * 100})

    df_sunburst = pd.DataFrame(data)

    fig = go.Figure(
        go.Sunburst(
            labels=df_sunburst["labels"],
            parents=df_sunburst["parents"],
            values=df_sunburst["values"],
            marker=dict(
                colorscale="Viridis", showscale=True, colorbar=dict(title="Value")
            ),
            textinfo="label+percent parent",
            hovertemplate="<b>%{label}</b><br>Value: %{value:.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Hierarchical Sunburst Chart", height=700, font=dict(size=11)
    )

    return fig


def create_treemap_viz(analyzer):
    """Treemap visualization"""
    data = []

    for category in analyzer.categories:
        top_terms = sorted(
            analyzer.results["category_tfidf"][category].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        for term, score in top_terms:
            data.append({"category": category, "term": term, "score": score * 1000})

    df_tree = pd.DataFrame(data)

    fig = px.treemap(
        df_tree,
        path=["category", "term"],
        values="score",
        color="score",
        color_continuous_scale="Viridis",
        title="Treemap: Term Importance",
    )

    fig.update_layout(height=700)

    return fig


def create_heatmap_correlation(analyzer):
    """Correlation heatmap"""
    categories = analyzer.categories
    n = len(categories)

    similarity_matrix = np.zeros((n, n))

    for i, cat1 in enumerate(categories):
        for j, cat2 in enumerate(categories):
            vec1 = [
                analyzer.results["category_tfidf"][cat1].get(f, 0)
                for f in analyzer.results["tfidf_features"]
            ]
            vec2 = [
                analyzer.results["category_tfidf"][cat2].get(f, 0)
                for f in analyzer.results["tfidf_features"]
            ]

            similarity_matrix[i][j] = cosine_similarity([vec1], [vec2])[0][0]

    fig = go.Figure(
        data=go.Heatmap(
            z=similarity_matrix,
            x=categories,
            y=categories,
            colorscale="RdBu",
            text=np.round(similarity_matrix, 2),
            texttemplate="%{text}",
            textfont={"size": 12},
            colorbar=dict(title="Similarity"),
        )
    )

    fig.update_layout(
        title="Category Similarity Heatmap",
        height=600,
        xaxis_title="Category",
        yaxis_title="Category",
    )

    return fig


def create_arc_diagram(analyzer):
    """Arc diagram"""
    cooccurrence = analyzer.results["cooccurrence"][:30]

    terms = list(set([term for pair, _ in cooccurrence for term in pair]))
    term_positions = {term: i for i, term in enumerate(terms)}

    fig = go.Figure()

    for (term1, term2), weight in cooccurrence:
        x1 = term_positions[term1]
        x2 = term_positions[term2]

        x_arc = np.linspace(x1, x2, 50)
        height = abs(x2 - x1) * 0.3
        y_arc = height * np.sin(np.linspace(0, np.pi, 50))

        fig.add_trace(
            go.Scatter(
                x=x_arc,
                y=y_arc,
                mode="lines",
                line=dict(color="rgba(100,100,200,0.3)", width=max(1, weight / 10)),
                showlegend=False,
                hovertemplate=f"{term1} ↔ {term2}<br>Count: {weight}<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=list(range(len(terms))),
            y=[0] * len(terms),
            mode="markers+text",
            marker=dict(size=12, color="#FF6B6B", line=dict(width=2, color="white")),
            text=terms,
            textposition="bottom center",
            textfont=dict(size=9),
            showlegend=False,
            hovertemplate="<b>%{text}</b><extra></extra>",
        )
    )

    fig.update_layout(
        title="Arc Diagram: Co-occurrence Patterns",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        height=600,
        plot_bgcolor="white",
    )

    return fig


def create_radial_tree(analyzer):
    """Radial tree"""
    G = analyzer.results["knowledge_graphs"]["main"]
    pos = nx.spring_layout(G, seed=42, k=3)

    categories = [n for n in G.nodes() if G.nodes[n].get("node_type") == "category"]
    terms = [n for n in G.nodes() if G.nodes[n].get("node_type") == "term"]

    fig = go.Figure()

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        fig.add_trace(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(color="rgba(150,150,150,0.3)", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    cat_x = [pos[n][0] for n in categories]
    cat_y = [pos[n][1] for n in categories]

    fig.add_trace(
        go.Scatter(
            x=cat_x,
            y=cat_y,
            mode="markers+text",
            marker=dict(size=40, color="#FF6B6B", line=dict(width=2, color="#C92A2A")),
            text=categories,
            textposition="middle center",
            textfont=dict(size=10, color="white"),
            name="Categories",
            hovertemplate="<b>%{text}</b><extra></extra>",
        )
    )

    term_x = [pos[n][0] for n in terms]
    term_y = [pos[n][1] for n in terms]

    fig.add_trace(
        go.Scatter(
            x=term_x,
            y=term_y,
            mode="markers+text",
            marker=dict(size=15, color="#4ECDC4", line=dict(width=1, color="#0B7285")),
            text=terms,
            textposition="top center",
            textfont=dict(size=8, color="#495057"),
            name="Terms",
            hovertemplate="<b>%{text}</b><extra></extra>",
        )
    )

    fig.update_layout(
        title="Radial Knowledge Graph",
        showlegend=True,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        height=800,
        plot_bgcolor="#f8f9fa",
    )

    return fig


# ============================================================================
# INTERACTIVE DASHBOARD
# ============================================================================


def create_interactive_dashboard(analyzer):
    """Create complete dashboard"""

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
        suppress_callback_exceptions=True,
    )

    colors = {
        "background": "#f8f9fa",
        "text": "#2C3E50",
        "primary": "#3498db",
        "secondary": "#2ecc71",
        "accent": "#e74c3c",
    }

    viz_categories = {
        "Key Terms & Importance": [
            (
                "wordcloud",
                "Word Cloud",
                "cloud",
                "Shows most frequent words in each category; size = frequency. Identifies dominant themes.",
            ),
            (
                "tagcloud",
                "Tag Cloud Bubbles",
                "tags",
                "Interactive bubble visualization where bubble size represents term frequency and importance.",
            ),
        ],
        "Relationships & Connections": [
            (
                "knowledge_graph_3d",
                "3D Knowledge Graph",
                "project-diagram",
                "Network of categories (red) and top terms (cyan); shows semantic connections between topics.",
            ),
            (
                "force_graph",
                "Force-Directed Graph",
                "atom",
                "Physics-based layout showing how terms cluster around categories naturally; proximity = semantic similarity.",
            ),
            (
                "sankey",
                "Sankey Flow",
                "stream",
                "Visualizes flow from news categories to important terms; width shows term importance per category.",
            ),
            (
                "arc",
                "Arc Diagram",
                "bezier-curve",
                "Shows which terms co-occur together in documents; arc height indicates co-occurrence frequency.",
            ),
        ],
        "Structure & Clusters": [
            (
                "tsne",
                "t-SNE Clustering",
                "braille",
                "Each point = one document, colored by category; proximity shows semantic similarity between articles.",
            ),
            (
                "umap",
                "Alternative t-SNE View",
                "project-diagram",
                "Same document clustering with different perspective; validates category separation and structure.",
            ),
            (
                "dendrogram",
                "Hierarchical Tree",
                "sitemap",
                "Tree showing how news categories group together by topic similarity; distance = dissimilarity.",
            ),
            (
                "sunburst",
                "Sunburst Chart",
                "sun",
                "Interactive hierarchy: inner ring = categories, outer ring = top terms. Click to zoom into categories.",
            ),
            (
                "treemap",
                "Treemap",
                "th",
                "Space-filling rectangles showing term importance; size = term weight, color = TF-IDF score.",
            ),
        ],
        "Advanced Analytics": [
            (
                "radial",
                "Radial Tree",
                "circle-nodes",
                "Hub-and-spoke layout with categories as hubs and surrounding terms; balanced view of all topics.",
            ),
            (
                "heatmap",
                "Similarity Heatmap",
                "th-large",
                "Color matrix showing semantic similarity between category pairs (red=similar, blue=different).",
            ),
        ],
    }

    app.layout = dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.H1(
                                        [
                                            html.I(className="fas fa-newspaper me-3"),
                                            "BBC News Advanced Analytics",
                                        ],
                                        className="display-4 text-center mb-3",
                                        style={"color": colors["text"]},
                                    ),
                                    html.P(
                                        "Interactive Text Mining & Knowledge Discovery",
                                        className="lead text-center text-muted mb-4",
                                    ),
                                    html.Hr(className="my-4"),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dbc.Card(
                                                        [
                                                            dbc.CardBody(
                                                                [
                                                                    html.H4(
                                                                        f"{len(analyzer.df):,}",
                                                                        className="text-primary",
                                                                    ),
                                                                    html.P(
                                                                        "Documents",
                                                                        className="text-muted mb-0",
                                                                    ),
                                                                ]
                                                            )
                                                        ],
                                                        className="text-center shadow-sm",
                                                    )
                                                ],
                                                width=3,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Card(
                                                        [
                                                            dbc.CardBody(
                                                                [
                                                                    html.H4(
                                                                        f"{len(analyzer.categories)}",
                                                                        className="text-success",
                                                                    ),
                                                                    html.P(
                                                                        "Categories",
                                                                        className="text-muted mb-0",
                                                                    ),
                                                                ]
                                                            )
                                                        ],
                                                        className="text-center shadow-sm",
                                                    )
                                                ],
                                                width=3,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Card(
                                                        [
                                                            dbc.CardBody(
                                                                [
                                                                    html.H4(
                                                                        f"{len(analyzer.results['tfidf_features']):,}",
                                                                        className="text-info",
                                                                    ),
                                                                    html.P(
                                                                        "Unique Terms",
                                                                        className="text-muted mb-0",
                                                                    ),
                                                                ]
                                                            )
                                                        ],
                                                        className="text-center shadow-sm",
                                                    )
                                                ],
                                                width=3,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Card(
                                                        [
                                                            dbc.CardBody(
                                                                [
                                                                    html.H4(
                                                                        "13+",
                                                                        className="text-warning",
                                                                    ),
                                                                    html.P(
                                                                        "Visualizations",
                                                                        className="text-muted mb-0",
                                                                    ),
                                                                ]
                                                            )
                                                        ],
                                                        className="text-center shadow-sm",
                                                    )
                                                ],
                                                width=3,
                                            ),
                                        ],
                                        className="mb-4",
                                    ),
                                ],
                                className="bg-white rounded-3 shadow-lg p-5 mb-4",
                            )
                        ]
                    )
                ]
            ),
            html.Div(
                [
                    html.Div(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H3(
                                                [
                                                    html.I(
                                                        className="fas fa-layer-group me-2"
                                                    ),
                                                    category,
                                                ],
                                                className="mb-4 text-secondary",
                                            )
                                        ],
                                        width=12,
                                    )
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Card(
                                                [
                                                    dbc.CardBody(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.I(
                                                                        className=f"fas fa-{icon} fa-3x mb-3 text-primary"
                                                                    ),
                                                                    html.H5(
                                                                        title,
                                                                        className="card-title",
                                                                    ),
                                                                    html.P(
                                                                        desc,
                                                                        className="card-text text-muted small",
                                                                    ),
                                                                    dbc.Button(
                                                                        "View",
                                                                        id={
                                                                            "type": "viz-btn",
                                                                            "index": viz_id,
                                                                        },
                                                                        color="primary",
                                                                        className="mt-2",
                                                                        size="sm",
                                                                    ),
                                                                ],
                                                                className="text-center",
                                                            )
                                                        ]
                                                    )
                                                ],
                                                className="h-100 shadow-sm hover-card",
                                            )
                                        ],
                                        width=12,
                                        lg=6,
                                        xl=4,
                                        className="mb-4",
                                    )
                                    for viz_id, title, icon, desc in viz_list
                                ]
                            ),
                        ]
                    )
                    for category, viz_list in viz_categories.items()
                ]
            ),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle(id="modal-title")),
                    dbc.ModalBody(
                        dcc.Loading(
                            id="loading-viz",
                            type="default",
                            children=html.Div(id="modal-content"),
                        )
                    ),
                    dbc.ModalFooter(
                        dbc.Button("Close", id="close-modal", className="ms-auto")
                    ),
                ],
                id="viz-modal",
                size="xl",
                scrollable=True,
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Hr(className="my-5"),
                            html.Div(
                                [
                                    html.H5("📚 Research References", className="mb-3"),
                                    html.Ul(
                                        [
                                            html.Li(
                                                "Salton & Buckley (1988) - TF-IDF weighting"
                                            ),
                                            html.Li(
                                                "Van der Maaten & Hinton (2008) - t-SNE visualization"
                                            ),
                                            html.Li(
                                                "McInnes et al. (2018) - UMAP projection"
                                            ),
                                            html.Li(
                                                "Newman (2006) - Network modularity"
                                            ),
                                            html.Li(
                                                "Blondel et al. (2008) - Community detection"
                                            ),
                                        ],
                                        className="text-muted small",
                                    ),
                                    html.P(
                                        [
                                            html.Strong("Stack: "),
                                            "Python 3.9, Plotly 5.14, Dash 2.9, NetworkX 3.1, Scikit-learn 1.2",
                                        ],
                                        className="text-muted small mt-3",
                                    ),
                                ],
                                className="bg-light rounded p-4",
                            ),
                        ]
                    )
                ]
            ),
        ],
        fluid=True,
        className="py-4",
        style={"backgroundColor": colors["background"]},
    )

    @app.callback(
        Output("viz-modal", "is_open"),
        Output("modal-title", "children"),
        Output("modal-content", "children"),
        [
            Input({"type": "viz-btn", "index": dash.dependencies.ALL}, "n_clicks"),
            Input("close-modal", "n_clicks"),
        ],
        [State("viz-modal", "is_open")],
        prevent_initial_call=True,
    )
    def toggle_modal(viz_clicks, close_click, is_open):
        triggered_id = ctx.triggered_id

        if triggered_id == "close-modal" or triggered_id is None:
            return False, "", ""

        if isinstance(triggered_id, dict):
            viz_type = triggered_id["index"]
        else:
            return False, "", ""

        try:
            if viz_type == "wordcloud":
                figs = []
                for cat in analyzer.categories:
                    fig = create_wordcloud_plotly(
                        analyzer.results["term_freq"][cat],
                        cat,
                        f"Word Cloud: {cat.upper()}",
                    )
                    figs.append(dcc.Graph(figure=fig))
                return True, "Interactive Word Clouds", html.Div(figs)

            elif viz_type == "tagcloud":
                figs = []
                for cat in analyzer.categories:
                    fig = create_tag_cloud_bubbles(
                        analyzer.results["term_freq"][cat],
                        cat,
                        f"Tag Cloud: {cat.upper()}",
                    )
                    figs.append(dcc.Graph(figure=fig))
                return True, "Bubble Tag Clouds", html.Div(figs)

            elif viz_type == "knowledge_graph_3d":
                fig = create_network_graph_3d(
                    analyzer.results["knowledge_graphs"]["main"], "3D Knowledge Graph"
                )
                return True, "3D Knowledge Graph", dcc.Graph(figure=fig)

            elif viz_type == "force_graph":
                fig = create_force_directed_graph(analyzer)
                return True, "Force-Directed Graph", dcc.Graph(figure=fig)

            elif viz_type == "sankey":
                fig = create_sankey_category_flow(analyzer)
                return True, "Sankey Flow", dcc.Graph(figure=fig)

            elif viz_type == "arc":
                fig = create_arc_diagram(analyzer)
                return True, "Arc Diagram", dcc.Graph(figure=fig)

            elif viz_type == "tsne":
                fig = create_tsne_scatter(analyzer)
                return True, "t-SNE Clustering", dcc.Graph(figure=fig)

            elif viz_type == "umap":
                fig = create_umap_scatter(analyzer)
                return True, "UMAP Projection", dcc.Graph(figure=fig)

            elif viz_type == "dendrogram":
                fig = create_dendrogram_plot(analyzer)
                return True, "Hierarchical Clustering", dcc.Graph(figure=fig)

            elif viz_type == "sunburst":
                fig = create_sunburst_hierarchy(analyzer)
                return True, "Sunburst Chart", dcc.Graph(figure=fig)

            elif viz_type == "treemap":
                fig = create_treemap_viz(analyzer)
                return True, "Treemap", dcc.Graph(figure=fig)

            elif viz_type == "radial":
                fig = create_radial_tree(analyzer)
                return True, "Radial Tree", dcc.Graph(figure=fig)

            elif viz_type == "heatmap":
                fig = create_heatmap_correlation(analyzer)
                return True, "Similarity Heatmap", dcc.Graph(figure=fig)

            else:
                return True, "Visualization", html.P("Not found")

        except Exception as e:
            return (
                True,
                "Error",
                html.Div(
                    [
                        html.H5("Error", className="text-danger"),
                        html.P(str(e), className="text-muted"),
                    ]
                ),
            )

    app.index_string = """
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>BBC News Analytics</title>
            {%favicon%}
            {%css%}
            <style>
                .hover-card {
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                }
                .hover-card:hover {
                    transform: translateY(-5px);
                    box-shadow: 0 10px 25px rgba(0,0,0,0.15) !important;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    """

    return app


# ============================================================================
# MAIN EXECUTION
# ============================================================================

# ============================================================================
# EXPOSE SERVER FOR RENDER
# ============================================================================
# Initialize globally so Gunicorn can find 'server'
print("[STARTUP] Initializing Analyzer for Production...")
analyzer = BBCNewsAnalyzerAdvanced()
analyzer.analyze_all()

app = create_interactive_dashboard(analyzer)
server = app.server  # Required for Render/Gunicorn

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" " * 25 + "BBC NEWS ANALYTICS")
    print("=" * 80)
    print("\n[OK] READY!")
    print("[INFO] Open: http://127.0.0.1:8050/")

    app.run_server(debug=False, host="127.0.0.1", port=8050)
