import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'CAR DETAILS.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe
print("First few rows of the dataset:")
print(df.head())

# Summary statistics
print("\nSummary statistics:")
print(df.describe(include='all'))

# Data Distribution Visualization
# Histogram for numerical features
numerical_cols = ['year', 'selling_price', 'km_driven']

plt.figure(figsize=(15, 5))
for i, col in enumerate(numerical_cols):
    plt.subplot(1, 3, i + 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Boxplot for numerical features
plt.figure(figsize=(15, 5))
for i, col in enumerate(numerical_cols):
    plt.subplot(1, 3, i + 1)
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

# Categorical Feature Analysis
categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner']

plt.figure(figsize=(20, 10))
for i, col in enumerate(categorical_cols):
    plt.subplot(2, 2, i + 1)
    sns.countplot(x=df[col])
    plt.title(f'Countplot of {col}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Violin plot of selling_price vs. categorical features
plt.figure(figsize=(20, 10))
for i, col in enumerate(categorical_cols):
    plt.subplot(2, 2, i + 1)
    sns.violinplot(x=df[col], y=df['selling_price'])
    plt.title(f'Selling Price vs. {col}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
