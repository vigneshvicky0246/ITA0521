import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset
file_path = r"C:\Users\tamil\Downloads\SR - Sheet1 (1).csv"
df = pd.read_csv(file_path)

# Assuming your CSV file has columns like 'measurement1', 'measurement2', ..., 'label'
# Adjust these column names based on your actual dataset

# Separate features (X) and labels (y)
X = df.drop('Radiation', axis=1)
y = df['Temperature']

# Convert categorical variables to numerical using one-hot encoding
X_encoded = pd.get_dummies(X, drop_first=True)

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.5, random_state=52)

# Step 3: Data Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train the SVM model
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train_scaled, y_train)

# Step 5: Make Predictions
y_pred = svm_model.predict(X_test_scaled)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Optionally, print more detailed classification report
print(classification_report(y_test, y_pred))
