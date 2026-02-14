import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

# 1️⃣ Load dataset
df = pd.read_csv("dataset/train.csv")

# Only use 'comment_text' and 'toxic'
X = df['comment_text']
y = df['toxic']

# 2️⃣ Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3️⃣ Convert text to numbers (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4️⃣ Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 5️⃣ Test accuracy
y_pred = model.predict(X_test_vec)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# 6️⃣ Save model and vectorizer
with open("toxic_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved!")
