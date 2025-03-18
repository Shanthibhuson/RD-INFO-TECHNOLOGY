import re
import random
import string
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def check_length(password):
    if len(password) < 8:
        return 1  # Weak password if it's less than 8 characters
    elif len(password) <= 12:
        return 2  # Medium strength
    else:
        return 3  # Strong password


def check_diversity(password):
    lower = re.search(r'[a-z]', password)
    upper = re.search(r'[A-Z]', password)
    digit = re.search(r'\d', password)
    special = re.search(r'[!@#$%^&*(),.?":{}|<>]', password)

    diversity_score = 0
    if lower: diversity_score += 1
    if upper: diversity_score += 1
    if digit: diversity_score += 1
    if special: diversity_score += 1
    
    return diversity_score


def check_predictability(password, common_passwords):
    if password.lower() in common_passwords:
        return 1  # Predictable password
    return 0  # Unpredictable password

# Machine Learning Model to classify password strength
def machine_learning_predict(password, model, vectorizer):
    features = vectorizer.transform([password])
    return model.predict(features)

# Example set of common passwords 
common_passwords = set([
    "password", "123456", "123456789", "qwerty", "abc123", "letmein", "welcome"
])


passwords = ['password123', 'Qwerty!123', 'P@ssw0rD1234', '123456', 'welcome@123']
strength = [0, 1, 2, 0, 1]  # Strength: 0=weak, 1=medium, 2=strong


def password_to_features(password):
    length_score = check_length(password)
    diversity_score = check_diversity(password)
    return [length_score, diversity_score]


features = [password_to_features(pwd) for pwd in passwords]
X_train = np.array(features)
y_train = np.array(strength)


model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)


class SimpleVectorizer:
    def transform(self, passwords):
        return np.array([password_to_features(pwd) for pwd in passwords])

vectorizer = SimpleVectorizer()


def evaluate_password(password):
    
    length_score = check_length(password)
    diversity_score = check_diversity(password)
    predictability_score = check_predictability(password, common_passwords)
    
    
    total_score = length_score + diversity_score - predictability_score
    
    print(f"Password: {password}")
    print(f"Length Score: {length_score} (Weak=1, Medium=2, Strong=3)")
    print(f"Diversity Score: {diversity_score} (1-4)")
    print(f"Predictability Score: {predictability_score} (Weak=1, Strong=0)")
    
    
    ml_prediction = machine_learning_predict(password, model, vectorizer)
    print(f"Machine Learning Prediction: {['Weak', 'Medium', 'Strong'][ml_prediction[0]]}")
    
    
    if total_score >= 6:
        strength = "Strong"
    elif total_score >= 3:
        strength = "Medium"
    else:
        strength = "Weak"
    
    print(f"Overall Password Strength: {strength}")

# Example usage
password = "Qwerty!123"
evaluate_password(password)

password = "password123"
evaluate_password(password)

password = "P@ssw0rD1234"
evaluate_password(password)
