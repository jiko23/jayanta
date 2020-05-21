import pandas as pd
from crossbox_1 import *

####### Reading the train and test datasets and describing them. ###############
train_df = pd.read_csv('Train_Set.csv',index_col = None,header=0)
train_df = train_df.sample(frac=1) ## SHUFFILING THE DATA FOR BETTER CLASSIFICATION
print("Train_Data_Description::",'\n',train_df.describe())

test_df = pd.read_csv('Test_Set.csv',index_col = None,header = 0)
print("Test_Data_Description::",'\n',test_df.describe())

######## Checking missing values in train dataset and handeling it. ############
missing_data = train_df.isnull().sum().sum()
print("Number of missing values in train_set::",missing_data)

if missing_data > 0 :
	train_df.fillna(train_df.median(),inplace=True)
	missing_data = train_df.isnull().sum().sum()
	print("After Handeling Missing Values in Train Set.Misiing values::",missing_data)

"""
As the dataset is already been normalized and scalled so no need to again
normalize and rescale. If we again rescale the data then the scale will be disturbed.
"""

########## Seperating features and label from train and test dataframes. ##############

train_label = train_df['Label'] ## train label
test_label = test_df['Label'] ## test label

"""
As per instructions 'account_id' and 'week_start_date' are not to be included for now
so, droping these two columns from dataset for now.
"""
train_df = train_df.drop(['account_id','week_start_date','Label'],axis=1)
test_df = test_df.drop(['account_id','week_start_date','Label'],axis=1)

train_feature_names = []
test_feature_names = []

for cols in train_df:
	train_feature_names.append(cols)

for cols in test_df:
	test_feature_names.append(cols)

####### Calling the model for predictions and stats. ##########
crossbox = Crossbox_Test(train_df,test_df,train_label,test_label,train_feature_names,test_feature_names)
crossbox.stack_model()