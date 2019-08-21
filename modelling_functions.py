# data prep
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PolynomialFeatures, MinMaxScaler
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

#Classification Models
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Regression Models
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# evaluation metrics
from scikitplot.metrics import plot_confusion_matrix, plot_precision_recall, plot_lift_curve, plot_roc
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_predict, RandomizedSearchCV, GridSearchCV


#Visualization 
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings('ignore')

def ml_pipeline(model,data,target,test_size = 0.2, drop_columns = [],returns = 'Model'):
    """
    This function can either return:
    1. The test score of one model
    2. The pipeline
    
    """
    #drop duplicates and other unwanted features
    data = data.drop_duplicates()
    to_drop = drop_columns + target
    x = data.drop(columns = to_drop)
    
    y = data[target].values
    
    #Splitting data into train and test without any preproprecessing 
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=test_size, random_state=1)
    
    # Most models expect y to be a 1d array
    y_train = y_train.reshape(len(y_train),)
    y_test = y_test.reshape(len(y_test),)

    # SimpleImputer to fill in any missing values
    # Standard Scaler to scale values 
    # OneHotEncoder to transform the categorical values into binary columns for each category
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])


    numeric_features = x.select_dtypes(include='number').columns
    categorical_features = x.select_dtypes(exclude='number').columns
    #ordinal_features = 

    #use the ColumnTransformer to apply the transformations to the correct columns in the dataframe.
    #(name, transformer, column(s)) tuples specifying the transformer objects to be applied to subsets of the data.
    preprocessor = (ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])
                   )

    model1 = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])
    if returns == 'Model':
        model1.fit(x_train, y_train)
        #fit model on train and print score on test set
        print(f"{model.__class__.__name__} : {model1.score(x_test, y_test)}")
    elif returns == 'Pipeline':
        return model1
    
def compare_algorithms(x, y, classification=True, cv=5):
    
    """
    This fuction 
    1. Helps with model selection
    2. Evaluates multiple models before hyper parameter tuning using cross validation
    
    x: features
    y: target
    classification: is this a classification or regression task
    
    For each model this function returns:
    1. Avergae score across test sets
    2. Box plot of all results across 
       showing the spread of the accuracy scores across each cross validation fold for each algorithm.
    
    """
    
    models = []
    
    # prepare models
    if classification == True:
        models.append(('LR', LogisticRegression()))
        #models.append(('LDA', LinearDiscriminantAnalysis()))
        #models.append(('KNN', KNeighborsClassifier()))
        models.append(('DT', DecisionTreeClassifier()))
        models.append(('RF', RandomForestClassifier()))
        #models.append(('SVM', SVC()))
        scoring = ['accuracy']
        
    else:
        models.append(('SVR', SVR()))
        models.append(('SVR', LinearRegression()))
        models.append(('SVR', Ridge()))
        models.append(('SVR', Lasso()))
        scoring = ['r2','neg_mean_absolute_error','neg_mean_squared_error']
        
    # evaluate each model in turn
    
    results = []
    names = []
    seed = 7
    
    #iterate through models and store results
    for name, model in models:
        kfold = model_selection.KFold(n_splits=cv, random_state=seed)
        cv_results = model_selection.cross_validate(model, x, y, cv=kfold, scoring=scoring)
        
        if classification == False:
            r2 = cv_results[f'test_{scoring[0]}']
            mae = cv_results[f'test_{scoring[1]}'].mean()
            mse = cv_results[f'test_{scoring[2]}'].mean()
            results.append(r2)
            msg = f"{name} -- R2 : {r2.mean()}, MAE : {mae}, MSE : {mse}"
            print(msg)
            
        else:
            ba = cv_results[f'test_{scoring[0]}']
            results.append(ba) 
            msg = f"{name} -- Accuracy : {ba.mean()}"
            print(msg)
            
        names.append(name)
        
    #algorithm comparison using boxplots
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    ax.set_xlabel('Model', fontsize=10, labelpad=20)
    ax.set_ylabel('Score', fontsize=12, labelpad=20)
    plt.show()
                 
        
def hp_tuning(model, params, data, target,test_size = 0.2, drop_columns = [],returns = 'Model'):
    """
    
    
    """
    #a = ml_pipeline(model,data,target=target,test_size = 0.2, drop_columns = drop_columns,returns = 'Pipeline')
    
    if  model.__class__.__name__ == 'RandomForestClassifier':
        
        params2 = {
         f'{a.steps[-1][0]}__n_estimators': params['n_estimators'],
         f'{a.steps[-1][0]}__max_depth':params['max_depth'],
         f'{a.steps[-1][0]}__max_features':params['max_features'],
         f'{a.steps[-1][0]}__min_samples_leaf':params['min_samples_leaf'],
         f'{a.steps[-1][0]}__min_samples_split': params['min_samples_split'],
        }
    
        gs = (GridSearchCV(estimator = model, 
                                    param_grid = params2, 
                                    scoring = 'accuracy', 
                                    cv = 5
                                   )
              )
    elif model.__class__.__name__ == 'DecisionTreeClassifier':
        
        params2 = {
         f'{a.steps[-1][0]}__max_depth':params['max_depth'],
         f'{a.steps[-1][0]}__max_features':params['max_features'],
         f'{a.steps[-1][0]}__min_samples_leaf':params['min_samples_leaf'],
         f'{a.steps[-1][0]}__min_samples_split': params['min_samples_split'],
        }
        
        gs = (GridSearchCV(estimator = model, 
                                    param_grid = params2, 
                                    scoring = 'accuracy', 
                                    cv = 5
                                   )
              )
        
        
    elif model.__class__.__name__ == 'LogisticRegression':
        
        params2 = {
            f'{a.steps[-1][0]}__penalty': params['penalty'],
            f'{a.steps[-1][0]}__C': params['C'],
        } 
        
        gs = (GridSearchCV(estimator = model, 
                                    param_grid = params2, 
                                    scoring = 'accuracy', 
                                    cv = 5
                                   )
             )
        
              
    elif model.__class__.__name__ == 'KNeighborsClassifier':
        
        params2 = {
            f'{a.steps[-1][0]}__n_neighbors': params['n_neighbors'],
            f'{a.steps[-1][0]}__algorithm': params['algorithm'],
        } 
        
        gs = (GridSearchCV(estimator = model, 
                                    param_grid = params2, 
                                    scoring = 'accuracy', 
                                    cv = 5
                                   )
             )
    elif model.__class__.__name__ == 'SVC':
        
        params2 = {
            f'{a.steps[-1][0]}__C': params['C'],
            f'{a.steps[-1][0]}__kernel': params['kernel'],
            f'{a.steps[-1][0]}__degree': params['degree']
        } 
        
        gs = (GridSearchCV(estimator = model, 
                                    param_grid = params2, 
                                    scoring = 'accuracy', 
                                    cv = 5
                                   )
             )
              
    gs.fit(x_train, y_train)
    y_pred = gs.predict(x_test)
    
    return accuracy_score(y_test, y_pred), gs.best_params_
    
def multiple_cms(models,x, y):
    
    """
    This fuction evaluates multiple classification models using 5 fold cross validation
    
    models: list of model instances (A max of 6 models)
    x: features
    y: target
    
    For each model this function returns:
    1. Its confusion matrix
    
    """

    
    if type(y_train) is pd.core.series.Series:
        labels = y.unique()

    elif type(y_train) is np.ndarray:
        labels = np.unique(y)

    else:
        raise TypeError('Y must be a series or a numpy array')
        
    n = len(models)
    
    
    plt.figure(figsize=(20,15))
    plt.suptitle("Confusion Matrix Comparison",fontsize=20)
    
    for increment,model in zip(range(n),models):
        #cross validation ends up predicting class for every sample as every fold is eventually used as a test set 
        y_pred = model_selection.cross_val_predict(model, x, y, cv=5)
        cm = confusion_matrix(y,y_pred,labels=labels)
        plt.subplot(2,3,1+increment)
        plt.title(f"{model.__class__.__name__} Confusion Matrix",pad=20)     
        sns.heatmap(cm,cbar=False,annot=True,cmap="Greens",fmt="d",xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted',fontsize=16, labelpad=20)
        plt.ylabel('True',fontsize=16, labelpad=17)
        
        #pad between suptitle and plots
        #w_pad, h_pad : (height/width) between edges of adjacent subplots.
    plt.tight_layout(pad=8, w_pad=5, h_pad=5)

def evaluate(model, x, y):
    
    """
    Evaluation function for one classification model
    
    model: one model
    x: features
    y: target
    
    Function returns the following plots:
    
    1. Precision-Recall Curve
    2. ROC Curves
    3. Confusion Matrix
    
    """
    #generate cross validation estimates which returns the predicted class of each sample in the data 
    #cross validation ends up predicting class for every sample as every fold is eventually used as a test set 
    y_pred = model_selection.cross_val_predict(model, x, y, cv=5)
    
    #For method=’predict_proba’, probabilities are returned instead of classes
    y_pred_proba = model_selection.cross_val_predict(model, x, y, cv=5, method='predict_proba')
    
    if type(y) is pd.core.series.Series:
        labels = y.unique()

    elif type(y) is np.ndarray:
        labels = np.unique(y)

    else:
        raise TypeError('Y must be a series or a numpy array')
        
    fig, axes = plt.subplots(ncols=3, nrows=1)
    fig.set_size_inches(22,7)
    fig.suptitle(f"{model.__class__.__name__}",fontsize=16) 
    
    axes = axes.ravel()
    plot_precision_recall(y, y_pred_proba, ax=axes[0])
    plot_roc(y, y_pred_proba, ax=axes[1])
    plot_confusion_matrix(y, y_pred, ax=axes[2], normalize=True, labels=labels)
    
def evaluate_multiple(models, x, y):
    
    """
    
    Evaluation function for multiple classification models
    
    models: a list of models
    x: features
    y: target
    
    
    Function returns the following plots:
    
    1. Precision-Recall Curve
    2. ROC Curves
    3. Confusion Matrix
    
    """
    
    for model in models:
        evaluate(model, x, y)

    