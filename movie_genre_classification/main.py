import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Load Dataset
df = pd.read_csv(
    "train_data.txt",
    sep=" ::: ",
    engine="python",
    names=["ID", "TITLE", "GENRE", "DESCRIPTION"]
)

# Show first rows
print(df.head())

# Input and Output
X = df["DESCRIPTION"]
y = df["GENRE"]

# Convert text into numbers
vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Custom Prediction
movie = ["A superhero saves the world from dangerous aliens"]

movie = vectorizer.transform(movie)

prediction = model.predict(movie)

print("\nPredicted Genre:", prediction)
