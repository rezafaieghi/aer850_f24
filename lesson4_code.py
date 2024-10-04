import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


df = pd.read_csv("data/housing.csv")
df = df.dropna()
df = df.reset_index(drop=True)
# print(df.info())

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
# Visualize the correlation matrix 
corr_matrix = (X_train.iloc[:,0:-5]).corr()
sns.heatmap(np.abs(corr_matrix))

# Based on inspection of the corr_matrix, determine colinear variables
# and explore their correlation with y.
# corr1 = y_train.corr(X_train['longitude'])
# print(corr1)
# corr2 = y_train.corr(X_train['latitude'])
# print(corr2)
# corr3 = y_train.corr(X_train['total_rooms'])
# print(corr3)
# corr4 = y_train.corr(X_train['total_bedrooms'])
# print(corr4)
# corr5 = y_train.corr(X_train['population'])
# print(corr5)
# corr6 = y_train.corr(X_train['households'])
# print(corr6)

#Drop colinear variables that has low correlation with y
X_train = X_train.drop(['longitude'], axis=1)
X_train = X_train.drop(['total_bedrooms'], axis=1)
X_train = X_train.drop(['population'], axis=1)
X_train = X_train.drop(['households'], axis=1)

X_test = X_test.drop(['longitude'], axis=1)
X_test = X_test.drop(['total_bedrooms'], axis=1)
X_test = X_test.drop(['population'], axis=1)
X_test = X_test.drop(['households'], axis=1)

#Double-check correlation matrix, make sure you have no colinear variables left
plt.figure()
corr_matrix_2 = (X_train.iloc[:,0:-5]).corr()
sns.heatmap(np.abs(corr_matrix_2))

#Linear Regression
linear_reg = LinearRegression()
param_grid_lr = {}  # No hyperparameters to tune for plain linear regression, but you still apply GridSearchCV.
grid_search_lr = GridSearchCV(linear_reg, param_grid_lr, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search_lr.fit(X_train, y_train)
best_model_lr = grid_search_lr.best_estimator_
print("Best Linear Regression Model:", best_model_lr)

#Support Vector Machine (SVM)
svr = SVR()
param_grid_svr = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto']
}
grid_search_svr = GridSearchCV(svr, param_grid_svr, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search_svr.fit(X_train, y_train)
best_model_svr = grid_search_svr.best_estimator_
print("Best SVM Model:", best_model_svr)

# Decision Tree
decision_tree = DecisionTreeRegressor(random_state=42)
param_grid_dt = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search_dt = GridSearchCV(decision_tree, param_grid_dt, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search_dt.fit(X_train, y_train)
best_model_dt = grid_search_dt.best_estimator_
print("Best Decision Tree Model:", best_model_dt)

# Random Forest
random_forest = RandomForestRegressor(random_state=42)
param_grid_rf = {
    'n_estimators': [10, 30, 50],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
grid_search_rf = GridSearchCV(random_forest, param_grid_rf, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)
best_model_rf = grid_search_rf.best_estimator_
print("Best Random Forest Model:", best_model_rf)

# Training and testing error for Linear Regression
y_train_pred_lr = best_model_lr.predict(X_train)
y_test_pred_lr = best_model_lr.predict(X_test)
mae_train_lr = mean_absolute_error(y_train, y_train_pred_lr)
mae_test_lr = mean_absolute_error(y_test, y_test_pred_lr)
print(f"Linear Regression - MAE (Train): {mae_train_lr}, MAE (Test): {mae_test_lr}")

# Training and testing error for SVM
y_train_pred_svr = best_model_svr.predict(X_train)
y_test_pred_svr = best_model_svr.predict(X_test)
mae_train_svr = mean_absolute_error(y_train, y_train_pred_svr)
mae_test_svr = mean_absolute_error(y_test, y_test_pred_svr)
print(f"SVM - MAE (Train): {mae_train_svr}, MAE (Test): {mae_test_svr}")

# Training and testing error for Decision Tree
y_train_pred_dt = best_model_dt.predict(X_train)
y_test_pred_dt = best_model_dt.predict(X_test)
mae_train_dt = mean_absolute_error(y_train, y_train_pred_dt)
mae_test_dt = mean_absolute_error(y_test, y_test_pred_dt)
print(f"Decision Tree - MAE (Train): {mae_train_dt}, MAE (Test): {mae_test_dt}")

# Training and testing error for Random Forest
y_train_pred_rf = best_model_rf.predict(X_train)
y_test_pred_rf = best_model_rf.predict(X_test)
mae_train_rf = mean_absolute_error(y_train, y_train_pred_rf)
mae_test_rf = mean_absolute_error(y_test, y_test_pred_rf)
print(f"Random Forest - MAE (Train): {mae_train_rf}, MAE (Test): {mae_test_rf}")