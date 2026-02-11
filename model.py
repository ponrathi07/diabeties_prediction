# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load the dataset
data = pd.read_csv("diabetes.csv")

# Select input features (X) and output (y)
X = data[["Glucose", "BloodPressure", "BMI", "Age"]]
y = data["Outcome"]

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the ML model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Save the trained model as model.pkl
pickle.dump(model, open("model.pkl", "wb"))

print("model.pkl created successfully")
