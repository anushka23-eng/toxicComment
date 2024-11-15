import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv('comments.csv')

# Display the first 5 rows of the dataset
print("Dataset Preview:")
print(df.head())

df.dropna(inplace=True)

# Display the count of each label
print("\nLabel Distribution:")
print(df['label'].value_counts())

X = df['comment']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

tfidf = TfidfVectorizer(max_df=0.7, stop_words='english')

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

pac = PassiveAggressiveClassifier(max_iter=50, random_state=42)
pac.fit(X_train_tfidf, y_train)

y_pred = pac.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f'\nModel Accuracy: {accuracy * 100:.2f}%')

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Plotting the confusion matrix using Matplotlib
fig, ax = plt.subplots(figsize=(6, 5))
cax = ax.matshow(conf_matrix, cmap='Blues')
fig.colorbar(cax)

ax.set_xticklabels([''] + ['Non-Toxic', 'Toxic'])
ax.set_yticklabels([''] + ['Non-Toxic', 'Toxic'])

for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
    ax.text(j, i, f'{conf_matrix[i, j]}',
            ha='center', va='center', color='white')

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

new_comments_df = pd.read_csv('comments.csv')
sample_comments = new_comments_df['comment'].tolist()

sample_features = tfidf.transform(sample_comments)

sample_predictions = pac.predict(sample_features)

# Print predictions for the sample comments (comments.csv)
print("\nSample Predictions:")
for comment, label in zip(sample_comments, sample_predictions):
    print(f"Comment: '{comment.strip()
                       }' --> {'Toxic' if label == 1 else 'Non-Toxic'}")
