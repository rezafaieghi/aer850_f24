import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import seaborn as sns


df = pd.read_csv("data/housing.csv")

print(df.info())

"""One Hote Encoding"""
# sklearn uses sparse data structures.
# my_encoder = OneHotEncoder()
# my_encoder.fit(df[['ocean_proximity']])
# encoded_data_sparse = my_encoder.transform(df[['ocean_proximity']])
# encoded_data = encoded_data.toarray()

# turn offf the sparse data option
my_encoder = OneHotEncoder(sparse_output=False)
my_encoder.fit(df[['ocean_proximity']])
encoded_data = my_encoder.transform(df[['ocean_proximity']])

category_names = my_encoder.get_feature_names_out()

encoded_data_df = pd.DataFrame(encoded_data, columns=category_names)
df = pd.concat([df, encoded_data_df], axis = 1)

df = df.drop(columns = 'ocean_proximity')


"""Data Splitting"""
#this is the very basic method of data splitting
# X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                     test_size = 0.2,
#                                                     random_state = 42)

#the use of stratified sampling is strongly recommended
df["income_categories"] = pd.cut(df["median_income"],
                          bins=[0, 2, 4, 6, np.inf],
                          labels=[1, 2, 3, 4])
my_splitter = StratifiedShuffleSplit(n_splits = 1,
                               test_size = 0.2,
                               random_state = 42)
for train_index, test_index in my_splitter.split(df, df["income_categories"]):
    strat_df_train = df.loc[train_index].reset_index(drop=True)
    strat_df_test = df.loc[test_index].reset_index(drop=True)
strat_df_train = strat_df_train.drop(columns=["income_categories"], axis = 1)
strat_df_test = strat_df_test.drop(columns=["income_categories"], axis = 1)



"""Variable Selection"""
# in this method of variable selection, we need to write down the name of all columns
# x_columns = ['longitude',
#              'latitude',
#              'housing_median_age',
#              'total_rooms',
#              'total_bedrooms',
#              'population',
#              'households',
#              'median_income',
#              'ocean_proximity']
# y_column = ['median_house_value']
# x_train = strat_df_train[x_columns] #indepent variables, predictor
# y = strat_df_train[y_column] #outcome measure, dependent variable, or label
# x_test = strat_df_train[x_columns]
# y_test = strat_df_train[y_column]

#in this method of variable selection, we just drop our outcome measure column from training data
X_train = strat_df_train.drop("median_house_value", axis = 1)
y_train = strat_df_train["median_house_value"]
X_test = strat_df_test.drop("median_house_value", axis = 1)
y_test = strat_df_test["median_house_value"]


"""Scaling"""
my_scaler = StandardScaler()
my_scaler.fit(X_train.iloc[:,0:-5])
scaled_data_train = my_scaler.transform(X_train.iloc[:,0:-5])
scaled_data_train_df = pd.DataFrame(scaled_data_train, columns=X_train.columns[0:-5])
X_train = scaled_data_train_df.join(X_train.iloc[:,-5:])

scaled_data_test = my_scaler.transform(X_test.iloc[:,0:-5])
scaled_data_test_df = pd.DataFrame(scaled_data_test, columns=X_test.columns[0:-5])
X_test = scaled_data_test_df.join(X_test.iloc[:,-5:])



"""Correlation Matrix"""
corr_matrix = (X_train.iloc[:,0:-5]).corr()
sns.heatmap(np.abs(corr_matrix))

corr1 = y_train.corr(X_train['longitude'])
print(corr1)
corr2 = y_train.corr(X_train['latitude'])
print(corr2)
corr3 = y_train.corr(X_train['total_rooms'])
print(corr3)
corr4 = y_train.corr(X_train['total_bedrooms'])
print(corr4)
corr5 = y_train.corr(X_train['population'])
print(corr5)
corr6 = y_train.corr(X_train['households'])
print(corr6)

X_train = X_train.drop(['longitude'], axis=1)
X_train = X_train.drop(['total_bedrooms'], axis=1)
X_train = X_train.drop(['population'], axis=1)
X_train = X_train.drop(['households'], axis=1)

X_test = X_test.drop(['longitude'], axis=1)
X_test = X_test.drop(['total_bedrooms'], axis=1)
X_test = X_test.drop(['population'], axis=1)
X_test = X_test.drop(['households'], axis=1)

"""Training Model"""
my_model1 = LinearRegression()
my_model1.fit(X_train, y_train)
y_pred_test1 = my_model1.predict(X_test)
mae1 = mean_absolute_error(y_test, y_pred_test1)
print(mae1)

my_model2 = RandomForestRegressor(n_estimators=30, random_state=42)
my_model2.fit(X_train, y_train)
y_pred_test2 = my_model2.predict(X_test)
mae2 = mean_absolute_error(y_pred_test2, y_test)
print(mae2)