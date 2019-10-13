#!/usr/bin/env python
# coding: utf-8

# 1. Import the dataset from uci.

# In[1]:


import pandas as pd
import numpy as np
dataframe1 = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data',header=None)


# 2. Check the datatypes of each column to distinguish the numeric and non-numeric data prior to preprocessing.

# In[2]:


def info(df):
    print("Data Types:")
    print(df.dtypes)
    print("Rows and Columns:")
    print(df.shape)
    print("Column Names:")
    print(df.columns)
    print("Null Values:")
    print(df.apply(lambda x: sum(x.isnull()) / len(df)))

info(dataframe1)


# 3. As a first step, let us select the features belonging to each category based on the datatypes. Also, let us drop the final result column '15'.

# In[3]:


numeric_features = dataframe1.select_dtypes(include=['int64', 'float64']).columns
categorical_features = dataframe1.select_dtypes(include=['object']).drop(dataframe1[[15]], axis=1).columns


# 4. Using iterative imputer for missing numerical values,most_frequent or mode as filler for missing categorical values as it is a good practice to avoid nulls in the data. OneHotEncoder transforms each unique value in the categorical columns into a new column containing a 0 or 1 based on whether the value is present or not.

# In[4]:


from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', IterativeImputer(max_iter=10, random_state=0)),
    ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[
    ('imputer',  SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
    ('scaler', StandardScaler())])


# 5. Concatenate both numeric and categorical/non-numeric columns and store it in preprocessor

# In[5]:


from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])


# 6. Applying 80-20 train-test split on the training data as we need to check how our model is predicting.

# In[6]:


from sklearn.model_selection import train_test_split
X = dataframe1.drop(dataframe1[[15]], axis=1)
y = dataframe1[[15]]

info(X)
info(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
#info(y_train)


# 7. We are creating a simple logistic regression model to predict.

# In[7]:


from sklearn.linear_model import LogisticRegression
lr = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(solver='lbfgs'))])


# 8. Fitting the model on the training data, and printing the accuracy score using the built in score method.

# In[8]:


lr.fit(X_train, y_train)
print("model score: %.3f" % lr.score(X_test, y_test))

