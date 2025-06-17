# Media Synergy: A Unified Recommendation System

## Project Overview

**Media Synergy** is a unified recommendation system that personalizes suggestions across **music**, **movies**, and **books**. By addressing challenges like the **long-tail problem** and **serendipitous discovery**, our system enhances user engagement across multiple domains in a seamless experience.

This cross-domain system applies a combination of **K-Means Clustering**, **Linear Regression**, and **Collaborative Filtering** to make intelligent content recommendations.

---

## Datasets Used

All datasets were sourced from **Kaggle**:

### Music (Spotify Dataset)
- Files: `data.csv`, `data_by_genres.csv`, `data_by_year.csv`, etc.
- Features: `danceability`, `valence`, `energy`, `genres`, `popularity`, `year`

### Movies (TMDb Dataset)
- File: `tmdb_5000_movies.csv`
- Features: `budget`, `revenue`, `genres`, `cast`, `crew`, `vote_count`, `overview`

### Books (Goodreads Dataset)
- Files: `books.csv`, `ratings.csv`
- Features: `original_title`, `author`, `average_rating`, `num_ratings`, `ISBN`

---

## Data Preprocessing

- Imputed missing values and handled `NaN`
- One-hot encoded categorical columns like `genre`
- Feature normalization (e.g., scaling `popularity`)
- Feature engineering: `song_age`, `decade`, `log-transformed` skewed data
- Used `ast.literal_eval` to clean stringified lists in movie data
- Applied **SMOTE** to address class imbalance

---

## Algorithms and Models

### Music Recommendation – *K-Means Clustering*
- Clustered songs and genres using `valence`, `energy`, `danceability`, etc.
- Dimensionality reduction with t-SNE and PCA
- Real-time genre-based song recommendations

### Movie Recommendation – *Linear & Logistic Regression*
- Modeled movie popularity and profitability
- Used genre input to predict and recommend top-rated movies
- Evaluated with MSE and R²

### Book Recommendation – *Collaborative Filtering*
- Merged books and ratings data
- Popularity-based top 50 book recommendations
- User-item matrix using pivot tables
- Generated personalized recommendations with cosine similarity

---

## Tech Stack & Tools

- **Language**: Python
- **Libraries**:  
  - Data: `pandas`, `numpy`, `ast`  
  - Visualization: `matplotlib`, `seaborn`, `plotly`, `yellowbrick`  
  - Machine Learning: `scikit-learn`, `SMOTE`  
  - Modeling: `KMeans`, `LinearRegression`, `RandomForest`, `pivot_table`

---

## Key Results

| Domain | Method | Key Insight |
|--------|--------|-------------|
| Music  | K-Means Clustering | Genre/song grouping improved diversity in recommendations |
| Movies | Regression | Genre-based movie selection driven by vote count and profitability |
| Books  | Collaborative Filtering | Personalized suggestions based on user similarity and popularity |

---

## Visual Highlights

- Genre and song clusters (t-SNE, PCA)
- Music feature trends over decades
- Top genres and books by popularity
- Error metrics for model evaluation

---

## Future Scope

- Integrate **hybrid models** to address cold-start issues
- Expand to include **TV shows**, **podcasts**, and **video games**
- Use **deep learning (autoencoders, transformers)** for better embeddings
- Implement user authentication for **profile-based recommendations**
- Add **privacy-preserving mechanisms** and fine-tuned **feedback loops**

---

## References

1. Christakou & Stafylopatis, *Hybrid Movie Recommender using Neural Networks*, IEEE, 2006  
2. Madhuri et al., *User-Based Collaborative Filtering for Book Recommendations*, IJARCCE, 2018  
3. Elbir et al., *Music Genre Classification with ML*, IEEE, 2018  

---

> 💡 *“Media Synergy isn’t just about recommendation. It’s about harmonizing the digital discovery experience across domains.”*
