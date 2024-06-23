import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib

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

# Define models to train
models = {
    "Linear Regression": LinearRegression(),
    "Bagging Classifier": BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42), n_estimators=10, random_state=42),
    "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Train each model and evaluate
best_model = None
best_score = float('-inf')

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if name == "Linear Regression":
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        score = r2
        print(f"{name} - Mean Squared Error: {mse}, R^2 Score: {r2}")
    else:
        accuracy = accuracy_score(y_test, y_pred)
        score = accuracy
        print(f"{name} - Accuracy: {accuracy:.2f}")
    
    if score > best_score:
        best_score = score
        best_model = model

print(f"Best model: {best_model.__class__.__name__} with score: {best_score}")

# Save the best model
model_filename = 'best_model.pkl'
joblib.dump(best_model, model_filename)
print(f"Saved the best model as {model_filename}")

# Load the best model
loaded_model = joblib.load(model_filename)
print("Loaded model:", loaded_model)

# Make a prediction with the loaded model (example)
sample_data = X_test[0].reshape(1, -1)
prediction = loaded_model.predict(sample_data)[0]
print(f"Prediction for sample data: {prediction}")


