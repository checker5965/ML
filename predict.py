# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import Imputer

# home_data=pd.read_csv('melb_data.csv')
# features = ['Rooms', 'Bathroom', 'Bedroom2', 'Landsize', 'Distance', 'Car', 'BuildingArea', 'YearBuilt']
# count = home_data.isnull().sum()
# imp_data = home_data.drop(['CouncilArea', 'Suburb', 'Address', 'Type', 'Method', 'SellerG', 'Date', 'Regionname'], axis=1)
# ccol = []
# for column in home_data.columns:
#     if  column != 'CouncilArea' and column != 'Suburb' and column != 'Address' and column != 'Type' and column != 'Method' and column != 'SellerG' and column != 'Date' and column != 'Regionname':
#         ccol.append(column)
# cleaner = Imputer()
# clean_data = pd.DataFrame(cleaner.fit_transform(imp_data))
# clean_data.columns = ccol

# y = clean_data.Price
# X = clean_data[features]
# Xtrain, Xtest, yTrain, yTest = train_test_split(X,y)
# model = RandomForestRegressor(random_state=1)
# model.fit(Xtrain, yTrain)
# predictions = model.predict(Xtest)
# print("The mean absolute error is: {}".format(mean_absolute_error(yTest, predictions)))


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

home_data=pd.read_csv('melb_data.csv')
home_data.dropna(axis=0, subset=['Price'], inplace=True)
del_data = home_data.drop(['CouncilArea'], axis=1)
count = del_data.isnull().sum()
# count = home_data.isnull().sum()
imp_data = home_data.drop(['CouncilArea', 'Suburb', 'Address', 'Type', 'Method', 'SellerG', 'Date', 'Regionname'], axis=1)
ccol = []
for column in home_data.columns:
    if  column not in ['CouncilArea', 'Suburb', 'Address', 'Type', 'Method', 'SellerG', 'Date', 'Regionname']:
        ccol.append(column)
cleaner = Imputer()
clean_data = pd.DataFrame(cleaner.fit_transform(imp_data))
clean_data.columns = ccol
data_without_impute = del_data.drop(ccol, axis=1)
data_without_impute = data_without_impute.drop(['Address', 'Date'], axis=1)
low_cardinality_cols = [col for col in data_without_impute.columns if data_without_impute[col].nunique()<10]
data_for_encoding = data_without_impute[low_cardinality_cols]
one_hot_encoded_data = pd.get_dummies(data_for_encoding)
feat_cols=[]
for col in one_hot_encoded_data.columns:
    feat_cols.append(col)
features = ['Rooms', 'Bathroom', 'Bedroom2', 'Landsize', 'Distance', 'Car', 'BuildingArea', 'YearBuilt'] + feat_cols
final_data = pd.concat([one_hot_encoded_data, clean_data], axis=1)
y = final_data.Price
X = final_data[features]
Xtrain, Xtest, yTrain, yTest = train_test_split(X,y)
model = RandomForestRegressor(random_state=1)
model.fit(Xtrain, yTrain)
predictions = model.predict(Xtest)
print("The mean absolute error is: {}".format(mean_absolute_error(yTest, predictions)))