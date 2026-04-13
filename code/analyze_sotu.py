"""
Full analysis pipeline for SOTU speeches (1950-2026).
Generates all figures for the LaTeX paper.

Analyses:
1. Speech length and vocabulary richness over time
2. TF-IDF partisan language analysis
3. LDA topic modeling with temporal evolution
4. Supervised classification (party prediction)
5. Word embedding (PCA/t-SNE) visualization
6. Sentiment and readability analysis
"""

import json
import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

warnings.filterwarnings('ignore')

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
FIG_DIR = os.path.join(BASE_DIR, '..', 'paper', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Style
sns.set_theme(style="whitegrid", font_scale=1.1)
PARTY_COLORS = {'Democrat': '#2166ac', 'Republican': '#b2182b'}
DECADE_CMAP = plt.cm.viridis


def load_corpus():
    """Load the SOTU corpus from JSON."""
    with open(os.path.join(DATA_DIR, 'sotu_corpus.json'), 'r') as f:
        speeches = json.load(f)
    print(f"Loaded {len(speeches)} speeches")

    # Build DataFrame
    rows = []
    stop_words = set(stopwords.words('english'))
    for s in speeches:
        text = s['text']
        words = word_tokenize(text.lower())
        words_alpha = [w for w in words if w.isalpha()]
        content_words = [w for w in words_alpha if w not in stop_words]
        sentences = sent_tokenize(text)

        rows.append({
            'year': s['year'],
            'president': s['president'],
            'party': s['party'],
            'text': text,
            'word_count': len(words_alpha),
            'unique_words': len(set(words_alpha)),
            'content_words': content_words,
            'sentence_count': len(sentences),
            'avg_sentence_len': np.mean([len(word_tokenize(sent)) for sent in sentences]) if sentences else 0,
            'type_token_ratio': len(set(words_alpha)) / max(len(words_alpha), 1),
            'decade': (s['year'] // 10) * 10
        })

    df = pd.DataFrame(rows)
    print(f"Corpus spans {df['year'].min()}-{df['year'].max()}, {df['party'].nunique()} parties")
    return df, speeches


# =============================================================================
# Figure 1: Speech Length and Vocabulary Richness Over Time
# =============================================================================
def fig1_speech_length_over_time(df):
    """Speech length trends with party coloring."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Word count over time
    ax = axes[0]
    for party, color in PARTY_COLORS.items():
        mask = df['party'] == party
        ax.scatter(df.loc[mask, 'year'], df.loc[mask, 'word_count'],
                   c=color, label=party, alpha=0.7, s=50, edgecolors='white', linewidth=0.5)

    # Rolling average
    df_sorted = df.sort_values('year')
    rolling = df_sorted.groupby('year')['word_count'].mean().rolling(5, min_periods=1).mean()
    ax.plot(rolling.index, rolling.values, 'k--', alpha=0.5, linewidth=1.5, label='5-year rolling avg.')
    ax.set_ylabel('Word Count')
    ax.set_title('State of the Union Address Length (1950–2026)')
    ax.legend(loc='upper right', fontsize=9)

    # Type-token ratio over time
    ax = axes[1]
    for party, color in PARTY_COLORS.items():
        mask = df['party'] == party
        ax.scatter(df.loc[mask, 'year'], df.loc[mask, 'type_token_ratio'],
                   c=color, label=party, alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
    rolling_ttr = df_sorted.groupby('year')['type_token_ratio'].mean().rolling(5, min_periods=1).mean()
    ax.plot(rolling_ttr.index, rolling_ttr.values, 'k--', alpha=0.5, linewidth=1.5)
    ax.set_ylabel('Type-Token Ratio')
    ax.set_xlabel('Year')
    ax.set_title('Vocabulary Richness (Type-Token Ratio)')

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig1_speech_length.pdf')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


# =============================================================================
# Figure 2: TF-IDF Partisan Language
# =============================================================================
def fig2_partisan_language(df):
    """Top distinctive words for each party using TF-IDF."""
    # Use per-speech TF-IDF, then average by party for more robust results
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english',
                            ngram_range=(1, 2), min_df=3, max_df=0.9)
    tfidf_matrix = tfidf.fit_transform(df['text'])
    feature_names = tfidf.get_feature_names_out()

    dem_mask = df['party'].values == 'Democrat'
    rep_mask = df['party'].values == 'Republican'
    dem_scores = np.asarray(tfidf_matrix[dem_mask].mean(axis=0)).flatten()
    rep_scores = np.asarray(tfidf_matrix[rep_mask].mean(axis=0)).flatten()

    # Differential TF-IDF: words more distinctive to each party
    diff = dem_scores - rep_scores

    n_top = 20
    dem_idx = np.argsort(diff)[-n_top:][::-1]
    rep_idx = np.argsort(diff)[:n_top]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Democrat distinctive words
    ax = axes[0]
    words_dem = [feature_names[i] for i in dem_idx]
    scores_dem = [diff[i] for i in dem_idx]
    ax.barh(range(n_top), scores_dem, color=PARTY_COLORS['Democrat'], alpha=0.8)
    ax.set_yticks(range(n_top))
    ax.set_yticklabels(words_dem, fontsize=9)
    ax.set_xlabel('Differential TF-IDF Score')
    ax.set_title('Democrat-Distinctive Language')
    ax.invert_yaxis()

    # Republican distinctive words
    ax = axes[1]
    words_rep = [feature_names[i] for i in rep_idx]
    scores_rep = [-diff[i] for i in rep_idx]
    ax.barh(range(n_top), scores_rep, color=PARTY_COLORS['Republican'], alpha=0.8)
    ax.set_yticks(range(n_top))
    ax.set_yticklabels(words_rep, fontsize=9)
    ax.set_xlabel('Differential TF-IDF Score')
    ax.set_title('Republican-Distinctive Language')
    ax.invert_yaxis()

    plt.suptitle('Partisan Language in SOTU Addresses (1950–2026)', fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig2_partisan_tfidf.pdf')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")

    return words_dem, words_rep


# =============================================================================
# Figure 3: LDA Topic Modeling
# =============================================================================
def fig3_topic_modeling(df, n_topics=8):
    """LDA topic model with temporal evolution of topics."""
    # Fit LDA
    count_vec = CountVectorizer(max_features=5000, stop_words='english',
                                min_df=3, max_df=0.9)
    doc_term = count_vec.fit_transform(df['text'])
    feature_names = count_vec.get_feature_names_out()

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42,
                                     max_iter=30, learning_method='batch')
    topic_dist = lda.fit_transform(doc_term)

    # Name topics by top words
    topic_labels = []
    top_words_per_topic = []
    for i, topic in enumerate(lda.components_):
        top_idx = topic.argsort()[-8:][::-1]
        words = [feature_names[j] for j in top_idx]
        top_words_per_topic.append(words)
        # Use top 3 words as label
        topic_labels.append(f"T{i+1}: {', '.join(words[:3])}")

    # Figure 3a: Topic composition heatmap by decade
    df_topics = pd.DataFrame(topic_dist, columns=[f'Topic {i+1}' for i in range(n_topics)])
    df_topics['decade'] = df['decade'].values
    decade_topics = df_topics.groupby('decade').mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(decade_topics.values.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_yticks(range(n_topics))
    ax.set_yticklabels(topic_labels, fontsize=8)
    ax.set_xticks(range(len(decade_topics)))
    ax.set_xticklabels([str(d) + 's' for d in decade_topics.index], fontsize=10)
    ax.set_xlabel('Decade')
    ax.set_title('Topic Prevalence by Decade (LDA, k=8)')
    plt.colorbar(im, ax=ax, label='Mean Topic Weight')
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig3a_topic_heatmap.pdf')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")

    # Figure 3b: Topic trends over time (stacked area)
    df_topics['year'] = df['year'].values
    yearly_topics = df_topics.groupby('year').mean().drop(columns=['decade'])

    fig, ax = plt.subplots(figsize=(10, 5))
    yearly_topics.plot.area(ax=ax, alpha=0.7, linewidth=0.5,
                            color=plt.cm.Set2(np.linspace(0, 1, n_topics)))
    ax.set_xlabel('Year')
    ax.set_ylabel('Topic Proportion')
    ax.set_title('Evolution of SOTU Topics Over Time')
    ax.legend(title='Topics', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig3b_topic_trends.pdf')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")

    return topic_labels, top_words_per_topic, topic_dist


# =============================================================================
# Figure 4: Party Classification Results
# =============================================================================
def fig4_classification(df):
    """Classify speeches by party using multiple models."""
    tfidf = TfidfVectorizer(max_features=3000, stop_words='english',
                            ngram_range=(1, 2), min_df=2)
    X = tfidf.fit_transform(df['text'])
    y = (df['party'] == 'Republican').astype(int).values

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        'Linear SVM': LinearSVC(max_iter=2000, C=1.0, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        results[name] = scores
        print(f"    {name}: {scores.mean():.3f} ± {scores.std():.3f}")

    # Figure 4a: Classification accuracy comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    model_names = list(results.keys())
    means = [results[m].mean() for m in model_names]
    stds = [results[m].std() for m in model_names]
    colors = ['#4393c3', '#92c5de', '#d6604d', '#f4a582']
    bars = ax.bar(model_names, means, yerr=stds, color=colors, alpha=0.85,
                  capsize=5, edgecolor='gray', linewidth=0.5)
    ax.set_ylabel('5-Fold Cross-Validation Accuracy')
    ax.set_title('Party Classification from SOTU Text')
    ax.set_ylim(0.4, 1.0)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance level')
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                f'{mean:.2f}', ha='center', fontsize=10, fontweight='bold')
    ax.legend()
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig4a_classification.pdf')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")

    # Figure 4b: Confusion matrix for best model (Logistic Regression)
    best_model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    best_model.fit(X, y)
    y_pred = best_model.predict(X)  # in-sample for confusion display
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Democrat', 'Republican'],
                yticklabels=['Democrat', 'Republican'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix (Logistic Regression, In-Sample)')
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig4b_confusion.pdf')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")

    # Get top features from logistic regression
    feature_names = tfidf.get_feature_names_out()
    coef = best_model.coef_[0]
    top_dem_idx = np.argsort(coef)[:15]
    top_rep_idx = np.argsort(coef)[-15:][::-1]
    top_dem_feats = [(feature_names[i], coef[i]) for i in top_dem_idx]
    top_rep_feats = [(feature_names[i], coef[i]) for i in top_rep_idx]

    return results, top_dem_feats, top_rep_feats


# =============================================================================
# Figure 5: Document Embeddings Visualization
# =============================================================================
def fig5_embeddings(df):
    """Visualize speeches in 2D embedding space using TF-IDF + t-SNE."""
    tfidf = TfidfVectorizer(max_features=3000, stop_words='english', min_df=2)
    X = tfidf.fit_transform(df['text'])

    # Reduce dimensionality first with SVD, then t-SNE
    svd = TruncatedSVD(n_components=30, random_state=42)
    X_svd = svd.fit_transform(X)

    tsne = TSNE(n_components=2, random_state=42, perplexity=15, max_iter=1000)
    X_2d = tsne.fit_transform(X_svd)

    # Figure 5a: Colored by party
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    for party, color in PARTY_COLORS.items():
        mask = df['party'] == party
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, label=party,
                   alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('SOTU Speeches by Party')
    ax.legend()

    # Figure 5b: Colored by decade
    ax = axes[1]
    decades = df['decade'].values
    unique_decades = sorted(df['decade'].unique())
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(unique_decades), vmax=max(unique_decades))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=decades, cmap=cmap,
                         alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
    cbar = plt.colorbar(scatter, ax=ax, label='Decade')
    cbar.set_ticks(unique_decades)
    cbar.set_ticklabels([f"{d}s" for d in unique_decades])
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('SOTU Speeches by Decade')

    plt.suptitle('t-SNE Projection of SOTU Addresses (TF-IDF Features)', fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig5_embeddings.pdf')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


# =============================================================================
# Figure 6: Readability and Sentence Complexity Over Time
# =============================================================================
def fig6_readability(df):
    """Average sentence length and readability metrics over time."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Average sentence length
    ax = axes[0]
    for party, color in PARTY_COLORS.items():
        mask = df['party'] == party
        ax.scatter(df.loc[mask, 'year'], df.loc[mask, 'avg_sentence_len'],
                   c=color, label=party, alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
    df_sorted = df.sort_values('year')
    rolling = df_sorted.groupby('year')['avg_sentence_len'].mean().rolling(5, min_periods=1).mean()
    ax.plot(rolling.index, rolling.values, 'k--', alpha=0.5, linewidth=1.5, label='5-year rolling avg.')
    ax.set_ylabel('Avg. Words per Sentence')
    ax.set_title('Sentence Complexity in SOTU Addresses')
    ax.legend(fontsize=9)

    # Unique words / total words by decade (boxplot)
    ax = axes[1]
    decade_data = []
    for _, row in df.iterrows():
        decade_data.append({
            'decade': str(row['decade']) + 's',
            'party': row['party'],
            'unique_pct': row['unique_words'] / max(row['word_count'], 1) * 100
        })
    decade_df = pd.DataFrame(decade_data)
    sns.boxplot(data=decade_df, x='decade', y='unique_pct', hue='party',
                palette=PARTY_COLORS, ax=ax, linewidth=0.8)
    ax.set_xlabel('Decade')
    ax.set_ylabel('Unique Words (%)')
    ax.set_title('Vocabulary Diversity by Decade and Party')
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig6_readability.pdf')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


# =============================================================================
# Figure 7: Keyword Frequency Trends
# =============================================================================
def fig7_keyword_trends(df):
    """Track frequency of key political terms over time."""
    keywords = {
        'Security & Defense': ['security', 'defense', 'military', 'terrorism', 'war', 'nuclear'],
        'Economy': ['economy', 'jobs', 'tax', 'budget', 'deficit', 'growth', 'inflation'],
        'Social Policy': ['education', 'health', 'healthcare', 'poverty', 'welfare', 'housing'],
        'Foreign Policy': ['allies', 'peace', 'diplomacy', 'trade', 'international', 'foreign'],
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()

    for idx, (category, words) in enumerate(keywords.items()):
        ax = axes[idx]
        # Calculate frequency per speech
        freqs = []
        for _, row in df.iterrows():
            text_lower = row['text'].lower()
            total_words = max(row['word_count'], 1)
            count = sum(text_lower.count(w) for w in words)
            freqs.append(count / total_words * 1000)  # per 1000 words

        df_temp = df[['year', 'party']].copy()
        df_temp['freq'] = freqs

        for party, color in PARTY_COLORS.items():
            mask = df_temp['party'] == party
            ax.scatter(df_temp.loc[mask, 'year'], df_temp.loc[mask, 'freq'],
                       c=color, alpha=0.5, s=30, label=party)

        # Rolling average
        yearly = df_temp.groupby('year')['freq'].mean()
        rolling = yearly.rolling(5, min_periods=1).mean()
        ax.plot(rolling.index, rolling.values, 'k-', alpha=0.7, linewidth=2)

        ax.set_title(category, fontsize=11)
        ax.set_ylabel('Frequency (per 1000 words)')
        if idx >= 2:
            ax.set_xlabel('Year')

    axes[0].legend(fontsize=8)
    plt.suptitle('Key Political Themes in SOTU Addresses Over Time', fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig7_keyword_trends.pdf')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


# =============================================================================
# Figure 8: Word Cloud Comparison
# =============================================================================
def fig8_wordclouds(df):
    """Word clouds for Democrat vs Republican speeches."""
    from wordcloud import WordCloud

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (party, color) in enumerate(PARTY_COLORS.items()):
        mask = df['party'] == party
        all_content = []
        for words_list in df.loc[mask, 'content_words']:
            all_content.extend(words_list)
        text = ' '.join(all_content)

        colormap = 'Blues' if party == 'Democrat' else 'Reds'
        wc = WordCloud(width=800, height=400, max_words=100,
                       background_color='white', colormap=colormap,
                       random_state=42).generate(text)

        ax = axes[idx]
        ax.imshow(wc, interpolation='bilinear')
        ax.set_title(f'{party} SOTU Language', fontsize=13)
        ax.axis('off')

    plt.suptitle('Most Frequent Content Words by Party (1950–2026)', fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig8_wordclouds.pdf')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print("SOTU ANALYSIS PIPELINE")
    print("=" * 60)

    print("\n[1/8] Loading corpus...")
    df, speeches = load_corpus()

    print("\n[2/8] Figure 1: Speech length and vocabulary richness...")
    fig1_speech_length_over_time(df)

    print("\n[3/8] Figure 2: Partisan language (TF-IDF)...")
    words_dem, words_rep = fig2_partisan_language(df)
    print(f"    Top Dem words: {words_dem[:5]}")
    print(f"    Top Rep words: {words_rep[:5]}")

    print("\n[4/8] Figure 3: Topic modeling (LDA)...")
    topic_labels, top_words, topic_dist = fig3_topic_modeling(df)
    for label in topic_labels:
        print(f"    {label}")

    print("\n[5/8] Figure 4: Party classification...")
    clf_results, top_dem_feats, top_rep_feats = fig4_classification(df)

    print("\n[6/8] Figure 5: Document embeddings (t-SNE)...")
    fig5_embeddings(df)

    print("\n[7/8] Figure 6: Readability analysis...")
    fig6_readability(df)

    print("\n[8/8] Figure 7: Keyword trends...")
    fig7_keyword_trends(df)

    print("\n[Bonus] Figure 8: Word clouds...")
    fig8_wordclouds(df)

    # Save summary statistics
    stats = {
        'n_speeches': len(df),
        'year_range': f"{df['year'].min()}-{df['year'].max()}",
        'dem_speeches': int((df['party'] == 'Democrat').sum()),
        'rep_speeches': int((df['party'] == 'Republican').sum()),
        'avg_word_count': float(df['word_count'].mean()),
        'classification_results': {name: {'mean': float(scores.mean()), 'std': float(scores.std())}
                                   for name, scores in clf_results.items()},
        'topic_labels': topic_labels,
    }
    stats_path = os.path.join(DATA_DIR, 'analysis_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved analysis stats to {stats_path}")

    print("\n" + "=" * 60)
    print("ALL FIGURES GENERATED SUCCESSFULLY")
    print("=" * 60)


if __name__ == '__main__':
    main()
