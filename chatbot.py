import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import pickle

# Load dataset
file_path = 'jobs.csv'  # Replace with actual file path
data = pd.read_csv(file_path)

# Ask user for the number of rows to use for training
num_rows = int(input(f"Enter the number of rows to use for training (max {len(data)}): "))
data = data.head(num_rows)

# Combine relevant columns into a single feature column
data['Features'] = (data['Key Skills'].fillna('') + ' ' +
                    data['Industry'].fillna('') + ' ' +
                    data['Job Experience Required'].fillna('') + ' ' +
                    data['Functional Area'].fillna('') + ' ' +
                    data['Role Category'].fillna('') + ' ' +
                    data['Job Salary'].fillna(''))

# Ensure the target column (Job Title) is valid
data['Job Title'] = data['Job Title'].fillna('Unknown')

# Split data
X = data['Features']  # Features
y = data['Job Title']  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),  # Convert text to feature vectors
    ('classifier', RandomForestClassifier(random_state=42))  # Classification model
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model for reuse
with open('career_recommender.pkl', 'wb') as model_file:
    pickle.dump(pipeline, model_file)

# Recommendation system
def recommend_career(skills, career, industry, experience, salary):
    # Combine inputs into a single feature string
    user_input = f"{skills} {career} {industry} {experience} {salary}"
    with open('career_recommender.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    prediction = model.predict([user_input])[0]
    return prediction

# Example usage with input examples
print("Provide your details to get a career recommendation:")
user_skills = input("Enter your skills (comma-separated, e.g., 'Python, Machine Learning, Data Analysis'): ")
user_career = input("Enter the career you're seeking (e.g., 'Data Scientist', 'Sales Executive', 'Testing Engineer'): ")
user_industry = input("Enter your preferred industry (e.g., 'IT-Software', 'Advertising', 'Real Estate'): ")
user_experience = input("Enter your years of experience (e.g., '0 - 1 yrs', '2 - 5 yrs', '5 - 10 yrs'): ")
user_salary = input("Enter your desired salary (e.g., '2,00,000 - 4,00,000 PA.', 'Not Disclosed'): ")

recommendation = recommend_career(user_skills, user_career, user_industry, user_experience, user_salary)
print(f"Recommended Career: {recommendation}")
