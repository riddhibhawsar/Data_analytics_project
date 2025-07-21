import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# STEP 1: Load Dataset
data = pd.read_csv('netflix_titles.csv')
print("Initial Data:")
print(data.head())

# STEP 2: Data Info
print("\nData Info:")
print(data.info())
print("\nNull values per column:")
print(data.isnull().sum())

# STEP 3: Data Cleaning
data.drop_duplicates(inplace=True)
data.dropna(subset=['director', 'country'], inplace=True)
data['date_added'] = pd.to_datetime(data['date_added'])
print("\nData after cleaning:")
print(data.info())

# STEP 4: Add date-related columns
data['year'] = data['date_added'].dt.year
data['month'] = data['date_added'].dt.month
data['day'] = data['date_added'].dt.day

# STEP 5: Content Type Distribution
type_counts = data['type'].value_counts()
plt.figure(figsize=(8, 5))
sns.barplot(x=type_counts.index, y=type_counts.values, palette='Set2')
plt.title('Distribution of Content by Type')
plt.xlabel('Type')
plt.ylabel('Count')
plt.show()

# STEP 6: Most Common Genres
data['genres'] = data['listed_in'].apply(lambda x: x.split(', '))
all_genres = sum(data['genres'], [])
genre_counts = pd.Series(all_genres).value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=genre_counts.values, y=genre_counts.index, palette='Set3')
plt.title('Most Common Genres on Netflix')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.show()

# STEP 7: Content Added Over Time
plt.figure(figsize=(12, 6))
sns.countplot(x='year', data=data, palette='coolwarm')
plt.title('Content Added Over the Years')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# STEP 8: Top 10 Directors
top_directors = data['director'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_directors.values, y=top_directors.index, palette='Blues_d')
plt.title('Top 10 Directors with the Most Titles')
plt.xlabel('Number of Titles')
plt.ylabel('Director')
plt.show()

# STEP 9: Word Cloud of Movie Titles
movie_titles = data[data['type'] == 'Movie']['title']
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(' '.join(movie_titles))
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Movie Titles')
plt.show()

# STEP 10: Top Countries with Most Content
top_countries = data['country'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_countries.values, y=top_countries.index, palette='pastel')
plt.title('Top 10 Countries with Most Netflix Content')
plt.xlabel('Count')
plt.ylabel('Country')
plt.show()

# STEP 11: Monthly Content Release
monthly_movies = data[data['type'] == 'Movie']['month'].value_counts().sort_index()
monthly_tv = data[data['type'] == 'TV Show']['month'].value_counts().sort_index()
plt.plot(monthly_movies.index, monthly_movies.values, label='Movies')
plt.plot(monthly_tv.index, monthly_tv.values, label='TV Shows')
plt.xlabel("Month")
plt.ylabel("Content Count")
plt.title("Monthly Content Releases")
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend()
plt.grid(True)
plt.show()
