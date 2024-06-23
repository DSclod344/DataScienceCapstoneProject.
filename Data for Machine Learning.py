
from pyexpat import model
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, r2_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error

# Load the dataset
file_path = 'CAR DETAILS.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe
print("First few rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing values in each column before handling:")
print(df.isnull().sum())

# Handle missing values by filling with the mean for numerical columns
df = df.fillna(df.mean(numeric_only=True))

# Separate features and target variable
X = df.drop('selling_price', axis=1)
y = df['selling_price']

# Identify categorical and numerical columns
categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner']
numerical_cols = ['year', 'km_driven']


# Preprocessing for numerical data: impute missing values and scale features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data: impute missing values and one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Preprocess the data
X_preprocessed = preprocessor.fit_transform(X)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Display the shapes of the resulting datasets
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

#Bagging
# Define base classifier (Decision Tree)
base_classifier = DecisionTreeClassifier(random_state=42)

# Define Bagging Classifier
bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10, random_state=42)

# Train the Bagging Classifier
bagging_classifier.fit(X_train, y_train)

# Predict on the test data
y_pred = bagging_classifier.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

#Ensemble techniques
# Define Gradient Boosting Classifier
boosting_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# Train the Gradient Boosting Classifier
boosting_classifier.fit(X_train, y_train)
# Predict on the test data
y_pred = boosting_classifier.predict(X_test)
# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Boosting Accuracy: {accuracy:.2f}')



