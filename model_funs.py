import pandas as pd 
from selenium import webdriver
import numpy as np 
from ReadData import *
import plotly.graph_objects as go
import matplotlib.pyplot as plt 
import seaborn as sns


pd.set_option("display.max_rows",
              10000, 
              "display.max_columns", 
              10000)



## SKLearn Modules we are going to need 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV 
from sklearn.svm import SVC 
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix 



## Start with the Random Forest Model
def RandomForestsModel(train_x, train_y): 
    '''Random Forest Model'''
    rf = RandomForestClassifier(random_state = 200) 
    rf.get_params()

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
           'max_features': max_features,
           'max_depth': max_depth,
           'min_samples_split': min_samples_split,
           'min_samples_leaf': min_samples_leaf,
           'bootstrap': bootstrap}

    cv = TimeSeriesSplit(n_splits = 5) 

    rf_random = RandomizedSearchCV(estimator = rf, 
                          param_distributions = random_grid, 
                          n_iter = 100, cv = cv, verbose = 2, n_jobs = -1)

    rf_random.fit(train_x, train_y) 
    return rf_random

## We also want an Support Vector Classifier to see which is most accurate
def SVM_Fit(train_x, train_y, kernel,
            params = [10**x for x in np.arange(-1,3,0.9)]): 
    '''Fit the SVM Machine given the kernel type, parameters, 
    data''' 
    
    if kernel == 'linear': 
        parameters = {'C': params} 
    else: 
        parameters = {'C': params, 
                     'gamma': params} 
    
    cv = TimeSeriesSplit(n_splits = 5) 
    
    model = GridSearchCV(estimator = SVC(kernel = kernel, probability = True), 
                        param_grid = parameters, 
                        cv = cv, 
                        verbose = 1) 
    
    model.fit(train_x, train_y) 
    return model




def Predict(fitted_model, test_x, test_y, 
           name):
    '''Prediction Accuracy'''
    prediction = fitted_model.predict(test_x) 
    score = accuracy_score(prediction, test_y) 
    prediction = pd.DataFrame({'prediction_{}'.format(name): prediction})
    print('The {} Model Score is: {}'.format(name, score)) 
    return prediction, score

## Run models 
def AllModels(x, y): 
        
    ## Radial SVC Model 
    rbf_model = SVM_Fit(x, y, 'rbf') 
    rbf_predict, rbf_score = Predict(rbf_model, x, y, 'radial')
    rbf_confusion = confusion_matrix(y,rbf_predict,normalize="true")
    
    rbf = {'model': rbf_model, 'predict': rbf_predict, 'score': rbf_score, 
          'confusion': rbf_confusion} 

    ## Linear SVC Model 
    lin_model = SVM_Fit(x, y, 'linear')
    lin_predict, lin_score = Predict(lin_model, x, y, 'linear')
    lin_confusion = confusion_matrix(y,lin_predict,normalize="true")
    
    lin = {'model': lin_model, 'predict': lin_predict, 'score': lin_score, 
      'confusion': lin_confusion} 

    ## Polynomial SVC Model
    poly_model = SVM_Fit(x, y, 'poly') 
    poly_predict, poly_score = Predict(poly_model, x, y, 'poly') 
    poly_confusion = confusion_matrix(y, poly_predict, normalize = 'true') 
    
    poly = {'model': poly_model, 'predict': poly_predict, 'score': poly_score, 
          'confusion': poly_confusion} 

    ## Sigmoid SVC Model 
    sig_model = SVM_Fit(x, y, 'sigmoid')
    sig_predict, sig_score = Predict(sig_model, x, y, 'sigmoid') 
    sig_confusion = confusion_matrix(y,sig_predict,normalize="true")
    
    sig = {'model': sig_model, 'predict': sig_predict, 'score': sig_score, 
          'confusion': sig_confusion} 

    ## Random Forest Model 
    rf_model = RandomForestsModel(x, y) 
    rf_predict, rf_score = Predict(rf_model, x, y, 'Random Forest') 
    rf_confusion = confusion_matrix(y,rf_predict,normalize="true")
    
    rf = {'model': rf_model, 'predict': rf_predict, 'score': rf_score, 
          'confusion': rf_confusion} 
    
    return rbf, lin, poly, sig, rf


def Confusion_Matrix(rbf, lin, poly, sig, rf): 
    fig, ax = plt.subplots(3, 2, figsize = (10,10)) 
    fig.suptitle('Confusion Matrices') 

    heatmap = lambda confusion, ax, model_name: sns.heatmap(confusion, ax = ax, annot = True, 
                                   xticklabels = [-1,0,1], 
                                   yticklabels = [-1,0,1] ).set_title(model_name)  

    heatmap(sig['confusion'], ax[0,0], 'Sigmoid')
    heatmap(lin['confusion'], ax[0,1], 'Linear') 
    heatmap(rbf['confusion'], ax[1,0], 'Radial') 
    heatmap(poly['confusion'], ax[1,1], 'Polynomial') 
    heatmap(rf['confusion'], ax[2,0], 'Random Forests')

    return fig 