import pandas as pd
import re
import nltk
import tldextract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK stopwords (only needed once)
nltk.download('stopwords')

# Feature Extraction Function
def extract_features(url):
    """Extract features from a URL for phishing detection."""
    features = {
        "url_length": len(url),
        "num_digits": sum(c.isdigit() for c in url),
        "num_special_chars": sum(c in "!@#$%^&*()_+=-{}[]|\\;:'\",.<>?/`~" for c in url),
        "num_subdomains": len(tldextract.extract(url).subdomain.split('.')),
        "contains_https": 1 if "https" in url else 0,
        "contains_ip_address": 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0
    }
    return features

# Load Dataset (Replace with actual dataset)
# Assume dataset.csv has 'url' and 'label' (1 = phishing, 0 = legitimate)
df = pd.read_csv("../data/phishing_dataset.csv")

# Extract text-based features
df["text_features"] = df["url"].apply(lambda x: " ".join(re.split('\W+', x.lower())))

# Extract numerical features
feature_df = df["url"].apply(extract_features).apply(pd.Series)

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=500)
text_features = vectorizer.fit_transform(df["text_features"]).toarray()

# Combine all features
X = pd.concat([feature_df, pd.DataFrame(text_features)], axis=1)
y = df["label"]

# Split into Train & Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression Model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Train Gradient Boosting Model
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)

# Evaluate Models
for model, name in zip([lr_model, gb_model], ["Logistic Regression", "Gradient Boosting"]):
    y_pred = model.predict(X_test)
    print(f"\n{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

print("Phishing URL detection training complete!")