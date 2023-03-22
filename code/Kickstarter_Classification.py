#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 19:32:51 2021

@author: hiradnourbakhsh
"""

################# Part I: Supervised Learning (Classification Model) #################

import pandas as pd

# importing dataset 

df = pd.read_excel('/Users/hiradnourbakhsh/Desktop/INSY 662/Individual Project/Kickstarter.xlsx')

################## Visualizing dataframe ########################
df.count()

df.columns

############# Data Preprocessing #################

# drop column which has almost entirely all missing values

df = df.drop(columns = ['launch_to_state_change_days'])

# remove disable_communication (unary variable)

df = df.drop(columns = ['disable_communication'])

# remove states other than successful or failed

df = df[(df.state != 'canceled') & (df.state != 'live') & (df.state != 'suspended')]

# remove deadline, state_changed_at, created_at, and launched_at variables
# it is possible that different months or different times of day would have different influence on project state
# as a result, we want to keep these variables separate from one another
# later on, we will use feature seletion techniques to determine which of these features are worthwhile to keep

df = df.drop(columns = ['deadline'])
df = df.drop(columns = ['state_changed_at'])
df = df.drop(columns = ['created_at'])
df = df.drop(columns = ['launched_at'])

# remove post-launch variables for classification task

df = df.drop(columns = ['goal','pledged', 'staff_pick', 'backers_count',  'usd_pledged',
                        'spotlight', 'deadline_weekday',
 'state_changed_at_weekday',
 'deadline_month',
 'deadline_day',
 'deadline_yr',
 'deadline_hr',
 'state_changed_at_month',
 'state_changed_at_day',
 'state_changed_at_yr',
 'state_changed_at_hr',
 'create_to_launch_days',
 'launch_to_deadline_days'])

# remove project_id and name

df=  df.drop(columns = ['project_id', 'name'])

# dummify country, dummify category, and currency variables

df = pd.get_dummies(df, columns = ['country'])
df = pd.get_dummies(df, columns = ['category'])
df = pd.get_dummies(df, columns = ['currency'])

# dummify weekday variables

df = pd.get_dummies(df, columns = ['created_at_weekday'])
df = pd.get_dummies(df, columns = ['launched_at_weekday'])

# dummify state

df = pd.get_dummies(df, columns = ['state'], drop_first = True)

# reorder columns to place state at index 0

mid = df.state_successful
df.drop(labels = ['state_successful'], axis = 1, inplace = True)
df.insert(0, 'state_successful', mid)
df

################## Classification Model: Gradient Boosting Algorithm ##################

############ GBT Cross Validation to determine optimal min_samples_split and n_estimators

X = df.iloc[:, 1:]

# y is our target variable - state
y = df['state_successful']

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 5)

for i in range(2,10):
    model = GradientBoostingClassifier(random_state = 0, min_samples_split = i, n_estimators = 100)
    scores = cross_val_score(estimator = model, X = X, y = y, cv = 5)
    print(i, ':', np.average(scores))
    
# best min_samples_split: 3

##### n_estimators

for i in range(1,9):
    model = GradientBoostingClassifier(random_state = 0, min_samples_split = 3, n_estimators = i*100)
    scores = cross_val_score(estimator = model, X = X, y = y, cv = 5)
    print(i, ':', np.average(scores))
    
# best n_estimators: 400

############### Building Gradient Boosting Model ################

X = df.iloc[:, 1:]

# y is our target variable - state
y = df['state_successful']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 5)

from sklearn.ensemble import GradientBoostingClassifier
gbt = GradientBoostingClassifier(random_state = 5, min_samples_split = 3, n_estimators = 400)

model = gbt.fit(X_train, y_train)
y_test_pred = model.predict(X_test)


######### Evaluating Gradient Boosting Task ###############

# Calculate accuracy score
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_test_pred)

# confusion matrix
metrics.confusion_matrix(y_test, y_test_pred)

print(pd.DataFrame(metrics.confusion_matrix(y_test, y_test_pred, labels=[0,1]), index=['true:0', 'true:1'], columns=['pred:0', 'pred:1']))

# Calculate precision

from sklearn import metrics
metrics.precision_score(y_test, y_test_pred)

# Calculate recall
metrics.recall_score(y_test, y_test_pred)

# Calculate F1 score
metrics.f1_score(y_test, y_test_pred)

################### Gradient Boosting Model Feature importances ###############

model.feature_importances_

pd.DataFrame(list(zip(X.columns,model.feature_importances_)), columns = ['predictor','feature importance'])

# dataframe containing model feature importances with index corresponding to column index in original dataframe
df_1 = pd.DataFrame(model.feature_importances_)

list(X.columns)[1] # name_len
list(X.columns)[11] # 'launched_at_yr'
list(X.columns)[47] # 'category_Plays'
list(X.columns)[50] # 'category_Software' - second highest importance score
list(X.columns)[55] # 'category_Web' - highest importance score

#### The most important features in the GBT model, using a threshold of 0.05, are: 
# name_len, launched_at_yr, category_Plays, category_Software, category_Web

############## GRADING #######################

############ Import Grading Data ###############

kickstarter_grading_df = pd.read_excel("Kickstarter-Grading.xlsx")

################# Preprocess Grading Data ##################

# drop column which has almost entirely all missing values

kickstarter_grading_df = kickstarter_grading_df.drop(columns = ['launch_to_state_change_days'])

# remove disable_communication (unary variable)

kickstarter_grading_df = kickstarter_grading_df.drop(columns = ['disable_communication'])

# remove states other than successful or failed

kickstarter_grading_df = kickstarter_grading_df[(kickstarter_grading_df.state != 'canceled') & (kickstarter_grading_df.state != 'live') & (kickstarter_grading_df.state != 'suspended')]

# remove deadline, state_changed_at, created_at, and launched_at variables
# it is possible that different months or different times of day would have different influence on project state
# as a result, we want to keep these variables separate from one another
# later on, we will use feature seletion techniques to determine which of these features are worthwhile to keep

kickstarter_grading_df = kickstarter_grading_df.drop(columns = ['deadline'])
kickstarter_grading_df = kickstarter_grading_df.drop(columns = ['state_changed_at'])
kickstarter_grading_df = kickstarter_grading_df.drop(columns = ['created_at'])
kickstarter_grading_df = kickstarter_grading_df.drop(columns = ['launched_at'])

# remove post-launch variables for classification task

kickstarter_grading_df = kickstarter_grading_df.drop(columns = ['goal','pledged', 'staff_pick', 'backers_count',  'usd_pledged',
                        'spotlight', 'deadline_weekday',
 'state_changed_at_weekday',
 'deadline_month',
 'deadline_day',
 'deadline_yr',
 'deadline_hr',
 'state_changed_at_month',
 'state_changed_at_day',
 'state_changed_at_yr',
 'state_changed_at_hr',
 'create_to_launch_days',
 'launch_to_deadline_days'])

# remove project_id and name

kickstarter_grading_df=  kickstarter_grading_df.drop(columns = ['project_id', 'name'])

# dummify country, dummify category, and currency variables

kickstarter_grading_df = pd.get_dummies(kickstarter_grading_df, columns = ['country'])
kickstarter_grading_df = pd.get_dummies(kickstarter_grading_df, columns = ['category'])
kickstarter_grading_df = pd.get_dummies(kickstarter_grading_df, columns = ['currency'])

# dummify weekday variables

kickstarter_grading_df = pd.get_dummies(kickstarter_grading_df, columns = ['created_at_weekday'])
kickstarter_grading_df = pd.get_dummies(kickstarter_grading_df, columns = ['launched_at_weekday'])

# dummify state

kickstarter_grading_df = pd.get_dummies(kickstarter_grading_df, columns = ['state'], drop_first = True)

# reorder columns to place state at index 0

mid = kickstarter_grading_df.state_successful
kickstarter_grading_df.drop(labels = ['state_successful'], axis = 1, inplace = True)
kickstarter_grading_df.insert(0, 'state_successful', mid)
kickstarter_grading_df

################## Setup the variables ####################
X_grading = kickstarter_grading_df.iloc[:, 1:]

# y is our target variable - state
y_grading = kickstarter_grading_df['state_successful']

################# Apply the model previously trained to the grading data ############3
y_grading_pred = model.predict(X_grading)

################### Calculate the accuracy score #################
accuracy_score(y_grading, y_grading_pred)



















