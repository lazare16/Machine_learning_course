import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the CSV file
data = pd.read_csv('teams_data.csv')

# Convert Wins and Losses columns to numeric for analysis
data['Wins'] = pd.to_numeric(data['Wins'], errors='coerce')
data['Losses'] = pd.to_numeric(data['Losses'], errors='coerce')
data['Year'] = pd.to_numeric(data['Year'], errors='coerce')

# Wins vs Losses Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(data['Wins'], data['Losses'], alpha=0.7, edgecolor='k')
plt.title('Wins vs Losses', fontsize=16)
plt.xlabel('Wins', fontsize=12)
plt.ylabel('Losses', fontsize=12)
plt.grid(True)
plt.show()

# Total Wins by Year Bar Plot
wins_by_year = data.groupby('Year')['Wins'].sum()
plt.figure(figsize=(12, 6))
wins_by_year.plot(kind='bar', color='skyblue')
plt.title('Total Wins by Year', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Total Wins', fontsize=12)
plt.xticks(rotation=45)
plt.show()

# Top 10 Teams with Most Wins
top_teams = data.groupby('Team Name')['Wins'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
top_teams.plot(kind='bar', color='orange')
plt.title('Top 10 Teams with Most Wins', fontsize=16)
plt.xlabel('Team Name', fontsize=12)
plt.ylabel('Total Wins', fontsize=12)
plt.xticks(rotation=45)
plt.show()

# Distribution of Wins Histogram
plt.figure(figsize=(10, 6))
plt.hist(data['Wins'], bins=20, color='purple', alpha=0.7)
plt.title('Distribution of Wins', fontsize=16)
plt.xlabel('Wins', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True)
plt.show()

# Line Plot of Wins and Losses Over the Years (Aggregate)
wins_losses_by_year = data.groupby('Year')[['Wins', 'Losses']].sum()
plt.figure(figsize=(12, 6))
wins_losses_by_year.plot(kind='line', marker='o')
plt.title('Wins and Losses Over the Years', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Total Count', fontsize=12)
plt.grid(True)
plt.legend(['Wins', 'Losses'], loc='upper left')
plt.show()

# ========================= MODEL CREATION AND TRAINING ========================= #
# Define a binary target: High Performer if Wins > 40, else Low Performer
data['High_Performer'] = (data['Wins'] > 40).astype(int)

# Features (X) and Target (y)
X = data[['Losses', 'Year']]  # Use 'Losses' and 'Year' as features
y = data['High_Performer']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Feature Importance
importances = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8, 5))
plt.bar(feature_names, importances, color='skyblue')
plt.title('Feature Importance', fontsize=16)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Importance', fontsize=12)
plt.grid(True)
plt.show()
