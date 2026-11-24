
import pandas as pd

# Load the Kaggle dataset
df = pd.read_csv("data/raw/feedback_student.csv", encoding="cp1252")

# Display first five records
print("FIRST FIVE RECORDS:")
print(df.head())

# Display column names
print("\nCOLUMN NAMES:")
print(df.columns.tolist())

# Display dataset dimensions
print("\nDATASET SHAPE:")
print(df.shape)

# Check missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# Check data types
print("\nDATA TYPES:")
print(df.dtypes)

# Display dataset information
print("\nDATASET INFORMATION:")
df.info()

# Display all unique quality labels
print("\nUNIQUE QUALITY LABELS:")
print(df["quality"].unique())

# Count records for each quality label
print("\nQUALITY DISTRIBUTION:")
print(df["quality"].value_counts())

# Check duplicate records
print("\nDUPLICATE RECORDS:")
print(df.duplicated().sum())

# Display sample comments
print("\nSAMPLE COMMENTS:")
for i in range(5):
    print("\nComment:", df["comment"].iloc[i])
    print("Quality:", df["quality"].iloc[i])


# Display all unique quality labels
print("\nUNIQUE QUALITY LABELS:")
print(df["quality"].unique())

# Count records for each quality label
print("\nQUALITY DISTRIBUTION:")
print(df["quality"].value_counts())

# Check duplicate records
print("\nDUPLICATE RECORDS:")
print(df.duplicated().sum())

# Display some sample comments
print("\nSAMPLE COMMENTS:")
for i in range(5):
    print(f"\nComment: {df['comment'].iloc[i]}")
    print(f"Quality: {df['quality'].iloc[i]}")

