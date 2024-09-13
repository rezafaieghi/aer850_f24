import numpy as np
import pandas as pd

# Create a Pandas dataframe
data = {
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50],
    'C': [100, 200, 300, 400, 500]
}
df = pd.DataFrame(data)

# Convert a Pandas dataframe to a NumPy array
array = df.to_numpy()
print("NumPy array:")
print(array)

# Perform mathematical operations on a NumPy array
result = np.square(array)
print("\nSquared values:")
print(result)

# Apply a NumPy function element-wise to a Pandas series
df['D'] = np.sqrt(df['C'])
print("\nModified dataframe:")
print(df)