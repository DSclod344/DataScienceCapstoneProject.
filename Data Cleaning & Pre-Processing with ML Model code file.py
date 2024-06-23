import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer

# Load the dataset
file_path = 'CAR DETAILS.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
print("First few rows of the dataset:")
print(df.head())

# Check for Missing Values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Check for Duplicates
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())
# Remove duplicate rows
df = df.drop_duplicates()

# Detect and Handle Outliers
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

# Detect outliers in 'selling_price' and 'km_driven'
selling_price_outliers = detect_outliers(df, 'selling_price')
km_driven_outliers = detect_outliers(df, 'km_driven')

print("\nOutliers in 'selling_price':")
print(selling_price_outliers)

print("\nOutliers in 'km_driven':")
print(km_driven_outliers)

# Remove outliers 
df = df[~df.index.isin(selling_price_outliers.index)]
df = df[~df.index.isin(km_driven_outliers.index)]

# Final dataset after cleaning
print("\nDataset after cleaning:")
print(df.head())

# Save the cleaned dataset
cleaned_file_path = 'CAR_DETAILS_CLEANED.csv'
df.to_csv(cleaned_file_path, index=False)
print(f"\nCleaned dataset saved to {cleaned_file_path}")


# Handle Missing Values
print("\nMissing values in each column before handling:")
print(df.isnull().sum())

# Remove rows with missing values
df_dropna = df.dropna()
print("\nMissing values after removing rows with missing values:")
print(df_dropna.isnull().sum())

# One-Hot Encoding for Categorical Variables
categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner']
df_encoded = pd.get_dummies(df, columns=categorical_cols)

print("\nData after one-hot encoding:")
print(df_encoded.head())

# Scaling Numerical Data
scaler = StandardScaler()
numerical_cols = ['year', 'selling_price', 'km_driven']




