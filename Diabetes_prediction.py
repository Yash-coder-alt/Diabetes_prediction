import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the diabetes dataset
diabetes_dataset = pd.read_csv("E:\\Systemtron\\diabetes.csv")

# Check if the dataset is loaded properly
if diabetes_dataset.empty:
    print("Dataset is empty or could not be loaded.")
    exit()  #Stop execution if the dataset is not loaded properly

# Display the first 5 rows
print(diabetes_dataset.head())

# Check the shape of the dataset (rows, columns)
print(f"Dataset Shape: {diabetes_dataset.shape}")

# Get statistical summary
print(diabetes_dataset.describe())

# Count of each outcome (0: non-diabetic, 1: diabetic)
print(diabetes_dataset['Outcome'].value_counts())

# Group data by 'Outcome' and get mean of each group
print(diabetes_dataset.groupby('Outcome').mean())

# Separate the features and target labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Store column names for later use
column_names = X.columns

# Print target labels
print(Y)

# Standardize the feature data
scaler = StandardScaler()
scaler.fit(X)  # Fit the scaler on the feature data
X = scaler.transform(X) # Transform X into a NumPy array

# Print standardized data
print(X)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

# Print the shapes of the datasets
print(f"Total Data Shape: {X.shape}")
print(f"Training Data Shape: {X_train.shape}")
print(f"Testing Data Shape: {X_test.shape}")

# Initialize the SVM classifier with a linear kernel
classifier = svm.SVC(kernel='linear')

# Train the SVM classifier on the training data
classifier.fit(X_train, Y_train)

# Model Evaluation - Accuracy on Training Data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print(f'Accuracy score of the training data: {training_data_accuracy:.2f}')

# Accuracy on Test Data
X_test_prediction = classifier.predict(X_test)  # Fix: Add this line
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print(f'Accuracy score of the test data: {test_data_accuracy:.2f}')

# Predict for a new input instance
input_data = [(1, 117, 88, 24, 145, 34.5, 0.403, 40),
              (6, 134, 80, 37, 370, 46.2, 0.238, 46),
              (13, 145, 82, 19, 110, 22.2, 0.245, 57),
              (3, 187, 70, 22, 200, 36.4, 0.408, 36)]
input_df = pd.DataFrame(input_data, columns=column_names)

# Convert input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Standardize the input data using the previously fitted scaler
std_data = scaler.transform(input_df)
print(std_data)

# Predict the outcome for the input data
predictions = classifier.predict(std_data.astype(float))

# Display the prediction result
for i, prediction in enumerate(predictions):
    if prediction == 0:
        print(f'Instance {i + 1}: The person is not diabetic.')
    else:
        print(f'Instance {i + 1}: The person is diabetic.')
