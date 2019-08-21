import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PolynomialFeatures
from sklearn.model_selection import train_test_split 
import warnings
#warnings.filterwarnings('ignore')

def input_output(data,target,test_size = 0.2, drop_columns = []):
    
    """ 
    A function which takes your dataset and:
    1. Replaces non-numeric columns with dummy columns 
    2. Scales numeric columns using a standard scaler
    3. Returns x_train, x_test, y_train, y_test
    
    dataset : dataframe which contains features and target column
    target : columns you are trying to predict, must be a list
    test_size : percentage of data you want to use for testing
    drop_columns : additional columns other than target column to drop from x_train/x_test 
    
    """
    ss = StandardScaler()
    
    # full list of columns we want to drop from x_train/x_test including target
    to_drop = drop_columns + target
    x = data.drop(columns = to_drop)
    
    #create dataset with dummy columns replacing non-numeric columns
    x = pd.get_dummies(x,columns = x.select_dtypes(exclude='number').columns).fillna(0)
    
    #scale x values
    x = ss.fit_transform(x)
    
    y = data[target].values
    
    #split the data into train and test
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1)
    
    # Most models expect y to be a 1d array
    y_train = y_train.reshape(len(y_train),)
    y_test = y_test.reshape(len(y_test),)
    
    return x_train, x_test, y_train, y_test