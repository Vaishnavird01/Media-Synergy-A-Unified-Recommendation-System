#Linear Regression on Movie Dataset
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import mean_squared_error

# Read the dataset
df = pd.read_csv('tmdb_5000_movies.csv')

# Preprocess the dataset
df['profitable'] = df.revenue > df.budget
df['profitable'] = df['profitable'].astype(int)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(how="any")

# Log transformation
for covariate in ['budget', 'popularity', 'runtime', 'vote_count', 'revenue']:
    df[covariate] = df[covariate].apply(lambda x: np.log10(1 + x))

# Separate positive revenue movies
positive_revenue_df = df[df["revenue"] > 0]

# Define targets and covariates
regression_target = 'revenue'
classification_target = 'profitable'
all_covariates = ['budget', 'popularity', 'runtime', 'vote_count', 'vote_average']

# User input for genres
genres_input = input("Enter genres (comma-separated): ")
genres = genres_input.split(',')

# Filter movies based on user-input genres
filtered_movies = df[df['genres'].str.contains('|'.join(genres))]

# Extract relevant features
features = filtered_movies[all_covariates].values

# Instantiate models
linear_regression = LinearRegression()
logistic_regression = LogisticRegression()
forest_regression = RandomForestRegressor(max_depth=4, random_state=0)
forest_classifier = RandomForestClassifier(max_depth=4, random_state=0)

# Train models
linear_regression.fit(positive_revenue_df[all_covariates], positive_revenue_df[regression_target])
logistic_regression.fit(positive_revenue_df[all_covariates], positive_revenue_df[classification_target])
forest_regression.fit(positive_revenue_df[all_covariates], positive_revenue_df[regression_target])
forest_classifier.fit(positive_revenue_df[all_covariates], positive_revenue_df[classification_target])

# Predictions
linear_regression_prediction = linear_regression.predict(features)
logistic_regression_prediction = logistic_regression.predict(features)
forest_regression_prediction = forest_regression.predict(features)
forest_classifier_prediction = forest_classifier.predict(features)

# Display results
print('\nMovie Recommended by Highest vote_count on various Algorithm models:')
print("\nRecommendation based on Linear Regression:")
print(filtered_movies[['original_title', 'vote_count']].assign(Linear_Regression_Prediction=linear_regression_prediction).sort_values(by='vote_count', ascending=False).head(1))

print("\nRecommendation based on Logistic Regression:")
print(filtered_movies[['original_title', 'vote_count']].assign(Logistic_Regression_Prediction=logistic_regression_prediction).sort_values(by='vote_count', ascending=False).head(1))

print("\nRecommendation based on Random Forest Regression:")
print(filtered_movies[['original_title', 'vote_count']].assign(Random_Forest_Regression_Prediction=forest_regression_prediction).sort_values(by='vote_count', ascending=False).head(1))

print("\nRecommendation based on Random Forest Classification:")
print(filtered_movies[['original_title', 'vote_count']].assign(Random_Forest_Classification_Prediction=forest_classifier_prediction).sort_values(by='vote_count', ascending=False).head(1))

# Calculate Mean Squared Error and R-squared for Linear Regression
linear_regression_mse = mean_squared_error(positive_revenue_df[regression_target], linear_regression.predict(positive_revenue_df[all_covariates]))
linear_regression_r2 = r2_score(positive_revenue_df[regression_target], linear_regression.predict(positive_revenue_df[all_covariates]))

# Calculate Mean Squared Error and R-squared for Logistic Regression
logistic_regression_mse = mean_squared_error(positive_revenue_df[classification_target], logistic_regression.predict(positive_revenue_df[all_covariates]))
logistic_regression_r2 = r2_score(positive_revenue_df[classification_target], logistic_regression.predict(positive_revenue_df[all_covariates]))

# Calculate Mean Squared Error and R-squared for Random Forest Regression
forest_regression_mse = mean_squared_error(positive_revenue_df[regression_target], forest_regression.predict(positive_revenue_df[all_covariates]))
forest_regression_r2 = r2_score(positive_revenue_df[regression_target], forest_regression.predict(positive_revenue_df[all_covariates]))

# Calculate Mean Squared Error and R-squared for Random Forest Classification
forest_classifier_mse = mean_squared_error(positive_revenue_df[classification_target], forest_classifier.predict(positive_revenue_df[all_covariates]))
forest_classifier_r2 = r2_score(positive_revenue_df[classification_target], forest_classifier.predict(positive_revenue_df[all_covariates]))

# Display results
print("\nPerformance Metrics:")
print(f"Linear Regression MSE: {linear_regression_mse}, R-squared: {linear_regression_r2}")
print(f"Logistic Regression MSE: {logistic_regression_mse}, R-squared: {logistic_regression_r2}")
print(f"Random Forest Regression MSE: {forest_regression_mse}, R-squared: {forest_regression_r2}")
print(f"Random Forest Classification MSE: {forest_classifier_mse}, R-squared: {forest_classifier_r2}")

#Linear Regression on Music Dataset
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import ast  # Import the ast module for literal_eval

# Load the dataset
data = pd.read_csv('data.csv')

# Convert the 'artists' column from string representation of a list to an actual list
data['artists'] = data['artists'].apply(ast.literal_eval)

# Explode the lists in the 'artists' column to separate rows for each artist
data_exploded = data.explode('artists')

# Group by song and calculate the mean popularity for each song
song_popularity = data_exploded.groupby('name')['popularity'].mean().reset_index()

# Define the features (X) and the target variable (y)
feature_names = ['valence', 'year', 'acousticness', 'danceability',
                 'duration_ms', 'energy', 'explicit', 'instrumentalness', 'key', 'liveness',
                 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']
X = data_exploded.groupby('name')[feature_names].mean()  # Take the mean of features for each song
y = data_exploded.groupby('name')['popularity'].mean()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Now, let's recommend one song based on the linear regression model
# Sort songs by predicted popularity in descending order
song_predictions = pd.DataFrame({'Song': X.index, 'Predicted Popularity': model.predict(X)})
sorted_songs = song_predictions.sort_values(by='Predicted Popularity', ascending=False)

# Select the top row as the recommended song
recommended_song = sorted_songs.head(1)

# Merge with the original dataset to get song names
recommended_song = pd.merge(recommended_song, song_popularity, left_on='Song', right_on='name', how='left')

# Display the recommended song along with its name and predicted popularity
print('\nMusic Recommended by Highest Predicted Popularity:')
print(recommended_song[['Song', 'name', 'Predicted Popularity', 'popularity']])

#Linear Regression on Book Dataset
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
books_data = pd.read_csv('books.csv')

# Explore and preprocess the data if needed
# For simplicity, assuming the 'average_rating' column is already available and cleaned


# Define features (X) and target variable (y)
feature_names = ['books_count', 'original_publication_year', 'ratings_count', 'work_ratings_count', 'work_text_reviews_count']
X = books_data[feature_names]
y = books_data['average_rating']

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Now, let's recommend books based on the linear regression model
# Sort books by predicted average rating in descending order
book_predictions = pd.DataFrame({'Book Title': books_data['title'], 'Predicted Average Rating': model.predict(X)})
sorted_books = book_predictions.sort_values(by='Predicted Average Rating', ascending=False)

# Display the recommended books along with their titles and predicted average ratings
# print('\nBooks Recommended by Highest Predicted Average Rating:')
# print(sorted_books[['Book Title', 'Predicted Average Rating']])

# ... (previous code)

# Now, let's recommend one book based on the linear regression model
# Sort books by predicted average rating in descending order
book_predictions = pd.DataFrame({'Book Title': books_data['title'], 'Predicted Average Rating': model.predict(X)})
sorted_books = book_predictions.sort_values(by='Predicted Average Rating', ascending=False)

# Select the top row as the recommended book
recommended_book = sorted_books.head(1)

# Display the recommended book along with its title and predicted average rating
print('\nBook Recommended by Highest Predicted Average Rating:')
print(recommended_book[['Book Title', 'Predicted Average Rating']])


#K-Means Clustering for Music Dataset
import os
import numpy as np
import pandas as pd

import seaborn as sns
import plotly.express as px 
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('data.csv')
genre_data = pd.read_csv('data_by_genres.csv')
year_data = pd.read_csv('data_by_year.csv')

print(data.info())

print(genre_data.info())

print(year_data.info())

from yellowbrick.target import FeatureCorrelation

feature_names = ['acousticness', 'danceability', 'energy', 'instrumentalness',
       'liveness', 'loudness', 'speechiness', 'tempo', 'valence','duration_ms','explicit','key','mode','year']

X, y = data[feature_names], data['popularity']

# Create a list of the feature names
features = np.array(feature_names)

# Instantiate the visualizer
visualizer = FeatureCorrelation(labels=features)

plt.rcParams['figure.figsize']=(20,20)
visualizer.fit(X, y)     # Fit the data to the visualizer
visualizer.show()

def get_decade(year):
    period_start = int(year/10) * 10
    decade = '{}s'.format(period_start)
    return decade

data['decade'] = data['year'].apply(get_decade)

sns.set(rc={'figure.figsize':(11 ,6)})
sns.countplot(data['decade'])

sound_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence']
fig = px.line(year_data, x='year', y=sound_features)
fig.show()

top10_genres = genre_data.nlargest(10, 'popularity')

fig = px.bar(top10_genres, x='genres', y=['valence', 'energy', 'danceability', 'acousticness'], barmode='group')
fig.show()

import sklearn
#print(sklearn.__version__)

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Create a KMeans object without specifying n_jobs in the constructor
kmeans = KMeans(n_clusters=10)

# Create the pipeline with StandardScaler and KMeans
cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', kmeans)])

# Select numeric features
X = genre_data.select_dtypes(np.number)

# Fit the pipeline
cluster_pipeline.fit(X)

# Assign the clusters to the 'cluster' column in genre_data
genre_data['cluster'] = cluster_pipeline.predict(X)

# Visualizing the Clusters with t-SNE

from sklearn.manifold import TSNE

tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=1))])
genre_embedding = tsne_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)
projection['genres'] = genre_data['genres']
projection['cluster'] = genre_data['cluster']

fig = px.scatter(
    projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'genres'])
fig.show()

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

song_cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=20, verbose=False))
])

X = data.select_dtypes(include=np.number)
number_cols = list(X.columns)

# Fit the pipeline
song_cluster_pipeline.fit(X)

# Predict and assign cluster labels
song_cluster_labels = song_cluster_pipeline.named_steps['kmeans'].labels_
data['cluster_label'] = song_cluster_labels

# Visualizing the Clusters with PCA

from sklearn.decomposition import PCA

pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
song_embedding = pca_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
projection['title'] = data['name']
projection['cluster'] = data['cluster_label']

fig = px.scatter(
    projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'title'])
fig.show()

# Assuming you want to recommend a song from a specific cluster, let's say cluster 0
recommended_song_cluster = 0

# Get a random song from the recommended cluster
recommended_song = data[data['cluster_label'] == recommended_song_cluster].sample(1)['name'].values[0]

print(f"Based on the analysis, I recommend you listen to '{recommended_song}' from Cluster {recommended_song_cluster}. Enjoy!")

#K-Means Clustering for Book Dataset
import os
import numpy as np
import pandas as pd

import seaborn as sns
import plotly.express as px 
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('books.csv')

print(data.info())

from yellowbrick.target import FeatureCorrelation

feature_names = ['book_id', 'best_book_id', 'work_id', 'books_count', 'work_ratings_count', 'work_text_reviews_count', 
                 'original_publication_year','ratings_count','ratings_1', 'ratings_2','ratings_3', 'ratings_4', 'ratings_5']

# Select features for clustering
X = data[feature_names]

# Replace NaN values with a default value (you can adjust this based on your data)
X = X.fillna(0)

# Check for infinite values and replace them with a default value
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

# Create a list of the feature names
features = np.array(feature_names)

# Instantiate the visualizer
visualizer = FeatureCorrelation(labels=features)

# Check for NaN values in the target variable 'average_rating'
if data['average_rating'].isnull().sum() > 0:
    print("Warning: There are missing values in the target variable. Consider handling them before clustering.")
else:
    # Fit the data to the visualizer
    plt.rcParams['figure.figsize'] = (20, 20)
    visualizer.fit(X, data['average_rating'])
    visualizer.show()
    
# Check for NaN values in 'original_publication_year'
if data['original_publication_year'].isnull().sum() > 0:
    
    data.dropna(subset=['original_publication_year'], inplace=True)

def get_decade(original_publication_year):
    if 1800 <= original_publication_year <= 2000:
        period_start = int(original_publication_year / 10) * 10
        decade = "{}'s".format(period_start)
        return decade
    else:
        return 'Unknown'
    
data['decade'] = data['original_publication_year'].apply(get_decade)

# Visualize the distribution of decades
sns.set(rc={'figure.figsize': (20, 20)})
sns.countplot(data['decade'])
plt.show()

from sklearn.impute import SimpleImputer

X, y = data[feature_names], data['average_rating']

# Select features for clustering
X = data[feature_names]

# Use SimpleImputer to handle NaN values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Create a KMeans object without specifying n_jobs in the constructor
kmeans = KMeans(n_clusters=10)

# Create the pipeline with StandardScaler and KMeans
cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', kmeans)])

# Fit the pipeline with the imputed data
cluster_pipeline.fit(X_imputed)

# Assign the clusters to the 'cluster' column in data
data['cluster'] = cluster_pipeline.predict(X_imputed)

import sklearn
#print(sklearn.__version__)

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE

tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=1))])
embedding = tsne_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=embedding)
projection['title'] = data['title']
projection['cluster'] = data['cluster']

fig = px.scatter(
    projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'title'])
fig.show()

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

books_cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=20, verbose=False))
])

X = data.select_dtypes(include=np.number)
number_cols = list(X.columns)

# Fit the pipeline
books_cluster_pipeline.fit(X_imputed)

# Predict and assign cluster labels
books_cluster_labels = books_cluster_pipeline.named_steps['kmeans'].labels_
data['cluster_label'] = books_cluster_labels

# Visualizing the Clusters with PCA

from sklearn.decomposition import PCA

pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
books_embedding = pca_pipeline.fit_transform(X_imputed)
projection = pd.DataFrame(columns=['x', 'y'], data=books_embedding)
projection['authors'] = data['authors']
projection['cluster'] = data['cluster_label']

fig = px.scatter(
    projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'authors'])
fig.show()

# Assuming you want to recommend a book from a specific cluster, let's say cluster 0
recommended_book_cluster = 0

# Get a random book from the recommended cluster
recommended_book = data[data['cluster_label'] == recommended_book_cluster].sample(1)['title'].values[0]

print(f"Based on the analysis, I recommend you read '{recommended_book}' from Cluster {recommended_book_cluster}. Happy reading!")

#K-Means Clustering for Movie Dataset
import os
import numpy as np
import pandas as pd

import seaborn as sns
import plotly.express as px 
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('tmdb_5000_movies.csv')

print(data.info())

from yellowbrick.target import FeatureCorrelation

feature_names = ['budget', 'id', 'revenue', 'runtime', 'vote_average', 'vote_count']

# Select features for clustering
X = data[feature_names]

# Replace NaN values with a default value (you can adjust this based on your data)
X = X.fillna(0)

# Check for infinite values and replace them with a default value
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

# Create a list of the feature names
features = np.array(feature_names)

# Instantiate the visualizer
visualizer = FeatureCorrelation(labels=features)

# Check for NaN values in the target variable 'average_rating'
if data['popularity'].isnull().sum() > 0:
    print("Warning: There are missing values in the target variable. Consider handling them before clustering.")
else:
    # Fit the data to the visualizer
    plt.rcParams['figure.figsize'] = (20, 20)
    visualizer.fit(X, data['popularity'])
    visualizer.show()
    

from sklearn.impute import SimpleImputer

X, y = data[feature_names], data['popularity']

# Select features for clustering
X = data[feature_names]

# Use SimpleImputer to handle NaN values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Create a KMeans object without specifying n_jobs in the constructor
kmeans = KMeans(n_clusters=10)

# Create the pipeline with StandardScaler and KMeans
cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', kmeans)])

# Fit the pipeline with the imputed data
cluster_pipeline.fit(X_imputed)

# Assign the clusters to the 'cluster' column in data
data['cluster'] = cluster_pipeline.predict(X_imputed)

import sklearn
#print(sklearn.__version__)

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE


tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=1))])
embedding = tsne_pipeline.fit_transform(X_imputed)
projection = pd.DataFrame(columns=['x', 'y'], data=embedding)
projection['popularity'] = data['genres']
projection['cluster'] = data['cluster']

fig = px.scatter(
    projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'popularity'])
fig.show()

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

movie_cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=20, verbose=False))
])

X = data.select_dtypes(include=np.number)
number_cols = list(X.columns)

# Fit the pipeline
movie_cluster_pipeline.fit(X_imputed)

# Predict and assign cluster labels
movie_cluster_labels = movie_cluster_pipeline.named_steps['kmeans'].labels_
data['cluster_label'] = movie_cluster_labels

# Visualizing the Clusters with PCA

from sklearn.decomposition import PCA
pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
movie_embedding = pca_pipeline.fit_transform(X_imputed)
projection = pd.DataFrame(columns=['x', 'y'], data=movie_embedding)
projection['title'] = data['original_title']
projection['cluster'] = data['cluster_label']

fig = px.scatter(
    projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'title'])
fig.show()

# Assuming you want to recommend a movie from a specific cluster, let's say cluster 0
recommended_movie_cluster = 0

# Get a random movie from the recommended cluster
recommended_movie = data[data['cluster_label'] == recommended_movie_cluster].sample(1)['original_title'].values[0]

print(f"Based on the analysis, I recommend you watch '{recommended_movie}' from Cluster {recommended_movie_cluster}. Enjoy the movie!")

#CONTENT BASED MOVIE RECOMMENDER SYSTEM
import numpy as np
import pandas as pd

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

movies.head()

credits.head()

movies = movies.merge(credits,on='title')
movies.head()
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

movies.head()

movies.isnull().sum()

movies.dropna(inplace=True)  # Drop null values

movies.duplicated().sum()  # Check for duplicates

# Preprocessing 
import ast

def convert(obj):
    List = []
    for i in ast.literal_eval(obj):
        List.append(i['name'])
    return List

movies['genres']= movies['genres'].apply(convert)

movies.head()

movies['keywords']= movies['keywords'].apply(convert)

movies.head()

import ast

def converts(obj):
    List = []
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
            List.append(i['name'])
            counter+=1
        else:
            break
    return List

movies['cast']=movies['cast'].apply(converts)

movies.head()

import ast

def fetch_director(obj):
    List = []
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            List.append(i['name'])
            break
    return List

movies['crew']=movies['crew'].apply(fetch_director)

movies.head()

movies['overview']=movies['overview'].apply(lambda x:x.split())

movies.head()

movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

movies.head()

movies['tags']=movies['overview'] + movies['genres']+movies['keywords']+movies['cast']+movies['crew']

movies.head()
movies.info()
new = movies[['movie_id','title','tags']]
new['tags']=new['tags'].apply(lambda x:" ".join(x))
new.head()
new['tags']=new['tags'].apply(lambda x:x.lower()) #Recommended to be in lower case
new.head()
new.info()
# Vectorization

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new['tags']).toarray()
vectors[0]
len(cv.get_feature_names())
cv.get_feature_names()
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)
sorted(list(enumerate(similarity[0])), reverse=True, key=lambda x:x[1])[1:6]
def recommend(movie):
    movie_index = new[new['title']== movie].index[0]
    distances=similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new.iloc[i[0]].title)
               
print("Top 5 movie recommendations based on the content is: ")
recommend('Batman Begins') 

#BOOK RECOMMENDER SYSTEM
import numpy as np
import pandas as pd
books=pd.read_csv('books.csv')
ratings=pd.read_csv('ratings.csv')
books.head()
ratings.head()
print(books.shape)
print(ratings.shape)
#Check for duplicates 
books.duplicated().sum()

#POPULARITY BASED
#Model Building
#Displaying the top 50 books with highest average rating
#Considering only those books which have a minimum of 50 ratings

ratings_with_name = ratings.merge(books, on='book_id')

ratings_with_name
num_rating_df= ratings_with_name.groupby('original_title').count()['rating'].reset_index()
num_rating_df
avg_rating_df= ratings_with_name.groupby('original_title').mean()['rating'].reset_index()
avg_rating_df
popularity_df= num_rating_df.merge(avg_rating_df, on='original_title')
popularity_df
popularity_df = popularity_df[popularity_df['rating_x']>=50].sort_values('rating_y', ascending=False).head(50)
popularity_df
popularity_df = popularity_df.merge(books,on='original_title')[['original_title','authors','rating_x','rating_y']]
popularity_df

#COLLABORATIVE FILTERING BASED
x=ratings_with_name.groupby('user_id').count()['rating'] >5
users = x[x].index
filtered_rating = ratings_with_name[ratings_with_name['user_id'].isin(users)]
y  = filtered_rating.groupby('original_title').count()['rating']>=50
famous_books = y[y].index
final_ratings= filtered_rating[filtered_rating['original_title'].isin(famous_books)]
pt = final_ratings.pivot_table(index='original_title', columns='user_id', values='rating')
pt
pt.fillna(0, inplace=True)
pt
from sklearn.metrics.pairwise import cosine_similarity
similarity_scores = cosine_similarity(pt)
similarity_scores.shape
def recommend(book_name):
    index = np.where(pt.index==book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x:x[1], reverse=True)[1:6]
    for i in similar_items:
        print(pt.index[i[0]])


print("Top 5 book recommendations based on the similarity is:")
recommend('A Bend in the Road')

