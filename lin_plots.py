#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 18:56:40 2021

@author: vijayyedidi


""" 


#%% 
import numpy as np 
import pandas as pd 
import plotly.graph_objects as go
from tqdm import tqdm

class PlotReg: 
    
    import pandas as pd 
    import numpy as np 
    import plotly.graph_objects as go
    from sklearn.linear_model import LinearRegression
    from tqdm import tqdm
    
    def __init__(self, df, name, 
                 dep_cols = [ 'b_brooklyn', 
                             'b_manhattan', 
                             'b_williamsburg', 
                             'b_queensboro',
                             'total']): 
    
            
            ## Define the dataframe 
            self.df = df 
            
            ## Define the dependent columns we're plotting 
            self.dep_cols = dep_cols 

            ## Define Name of Plot 
            self.name = name
            
            ## Define the independent columns we're plotting
            self.ind_cols = self.df.drop(columns = self.dep_cols,
                                    axis = 1).columns.to_list() 


            
            
    def SmallDF(self, ind, dep): 
        
        ## Create smal df with just dependent and independent variable
        small_df = self.df[[ind, dep]] 
        
        ## Order by the independent variable
        small_df = small_df.sort_values(by = ind, 
                                        ascending = True, 
                                        na_position = 'last') 
        
        return small_df 
    
    
    def FindPrediction(self, ind, dep): 
        

        
        ## Given the small df create prediction and 
        ## Return coefficients, along with prediction. 
        
        small_df = self.SmallDF(ind, dep) ## Create small df 
        
        ## Generate prediction 
        lm = self.LinearRegression(
            ).fit(
                small_df[ind].to_numpy().reshape(-1, 1), 
                small_df[dep].to_numpy().reshape(-1, 1))
        
        accuracy = lm.score(small_df[ind].to_numpy().reshape(-1, 1), 
                small_df[dep].to_numpy().reshape(-1, 1))
        
        
        intercept = lm.intercept_ 
        coef = lm.coef_ 
        
        ## Create Linear Regression Line 
        x = np.arange(small_df[ind].min(), 
                      small_df[ind].max(), 
                      0.001) 
        
        pred_y = x * coef + intercept
        pred_y = pred_y.flatten() 
        

        
        return coef, intercept, accuracy, x, pred_y, small_df
    
    
    def Plot(self): 
        
        
        ## Given independent, dependent, and prediction, generate plots 
        
        traces = []
        buttons  = [] 
        coefs = [] 
        intercepts = [] 
        accuracys = [] 
        names = [] 
        
        
        i = 0
        
        for ind in tqdm(self.ind_cols): 
            visibility_matrix = self.Visible(i) 
            buttons.append(self.FormatButton(ind, visibility_matrix)) 
            
            i += 1
            for dep in tqdm(self.dep_cols): 
                
                coef, intercept, accuracy, x, pred_y, small_df = self.FindPrediction(ind, dep)
            
                
                name = '{} vs {}'.format(ind, dep)
                print('starting {}. \n Slope = {}, Intercept = {}'.format(name, coef[0][0], intercept[0]) )
                
                
                ## For each independent column create a plot 
                ## For each dependent column 
                
                traces.append(
                    go.Scatter(
                        x = small_df[ind], 
                        y = small_df[dep], 
                        name = '{} Real'.format(name), 
                        mode = 'markers') ) 
                
                ##Prediction Plot 
                
                traces.append(
                    go.Scatter(
                        x = x, 
                        y = pred_y, 
                         
                        name = '{} Predicted'.format(name))) 
                
                coefs.append(coef[0][0])
                intercepts.append(intercept[0] )
                accuracys.append(accuracy) 
                names.append(name) 
                

                
                
                print('{} done'.format(name))

        metrics = pd.DataFrame({'name': names, 
                       'coef': coefs, 
                       'intercept': intercepts, 
                       'accuracy': accuracys}) 
            

        print('Traces complete. Compiling Graph') 



        fig = go.Figure() 
        fig.add_traces(traces) 
            
        fig.update_layout(updatemenus = [go.layout.Updatemenu(
                active = 0, 
                buttons = list(buttons))])
            
        print('Graph done. Formatting Metrics') 
            
        
                
        return fig, metrics
    
    def Visible(self, plot_num) :
        
        ## How many traces per plot 
        traces = len(self.dep_cols) * 2

        ## How many drop down plots
        plots = len(self.ind_cols) 
 

        show_plots = np.zeros((plots, traces), dtype = bool)
        
        
        show_plots[plot_num] = True
        
        show_plots = show_plots.flatten()
        
        
        return show_plots
    
    
    def FormatButton(self, ind_var, visible): 
        
        button = dict(label = ind_var, 
                      method = 'update', 
                      args = [{'visible': visible}, 
                              {'title': 'Total Traffic Based on {}. {}'.format(ind_var, self.name), 
                              'showlegend': True}])
        
        return button
        
        

    
        
        
        
        

            
            
            


        
 