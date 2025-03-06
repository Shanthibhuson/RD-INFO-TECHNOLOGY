import re

def check_password_strength(password):
    """Evaluate password strength based on rule-based criteria."""
    length = len(password) >= 12
    has_upper = bool(re.search(r'[A-Z]', password))
    has_lower = bool(re.search(r'[a-z]', password))
    has_digit = bool(re.search(r'\d', password))
    has_special = bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password))

    score = sum([length, has_upper, has_lower, has_digit, has_special])

    if score == 5:
        return "Strong"
    elif score >= 3:
        return "Moderate"
    else:
        return "Weak"

# Example Usage
password = input("Enter a password: ")
print(f"Password Strength: {check_password_strength(password)}")