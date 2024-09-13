import pandas as pd

# Reading a CSV file
csv_file = 'data.csv'
df = pd.read_csv(csv_file)

# Displaying the first few rows of the DataFrame
print("First few rows of the DataFrame:")
print(df.head())

# Getting basic information about the DataFrame
print("\nDataFrame info:")
print(df.info())

# Accessing columns and rows
print("\nSelecting a specific column:")
column_data = df['Column_Name']
print(column_data)

print("\nSelecting a specific row:")
row_data = df.loc[2]
print(row_data)

# Data aggregation and summary statistics
print("\nSummary statistics:")
print(df.describe())

# Data filtering and conditional selection
print("\nFiltering the DataFrame based on a condition:")
filtered_data = df[df['Column_Name'] > 10]
print(filtered_data)

# Sorting the DataFrame
print("\nSorting the DataFrame by a column:")
sorted_data = df.sort_values('Column_Name')
print(sorted_data)

# Adding a new column
df['New_Column'] = df['Column_Name'] * 2
print("\nDataFrame with a new column:")
print(df)

# Grouping and aggregation
grouped_data = df.groupby('Group_Column').mean()
print("\nGrouped DataFrame with mean values:")
print(grouped_data)
