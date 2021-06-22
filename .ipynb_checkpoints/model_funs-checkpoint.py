import pandas as pd 
import numpy as np 
from tqdm import tqdm 
import selenium
import csv
import os
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


## Prediction and accuracy function for different Machine Learning Models 
def Predict(fitted_model, test_x, test_y, 
           name):
    '''Prediction Accuracy'''
    prediction = fitted_model.predict(test_x) 
    score = accuracy_score(prediction, test_y) 
    prediction = pd.DataFrame({'prediction_{}'.format(name): prediction})
    print('The {} Model Score is: {}'.format(name, score)) 
    return prediction, score

## Read in input data from csv files 
def Read_Input(): 
    ## Read in input data from csv files 


    dfs = {}
    input_directory = os.path.join(os.getcwd(), 'csv_files/input_files') 

    ## Reads in all data in the input_files location
    try: 
        for root, dirs, files in os.walk(input_directory): 
            for file in files: 
                if file.endswith('.csv'): 
                    name = os.path.basename(file).split('.')
                    loc = os.path.join(input_directory, file) 
                    dfs[name[0]] = pd.read_csv(loc,index_col = 0, encoding = 'latin-1')  
    except: 
        print('reading files failed') 
                
    return dfs 


def Normalize_Team_Data(df): 
    '''Normalize team data''' 
    
    df_group = df.drop(columns = ['playoffs'])
    df_group = df_group.groupby(['year'])
    df_group = df_group.transform(lambda x: (x - x.mean())/(x.std())) 
    #df_group = df_group.drop(columns = ['level_0', 'index'])
    df_group[df_group.columns] = MinMaxScaler().fit_transform(df_group) 

    df = pd.concat(
        [df[['Team', 'year']], df_group, df['playoffs']], axis = 1
    )

    return df
    
    

def FormatTeamStats(team_opponent): 
    
    ## drop index, the games and minutes columns are not needed. Also drop nulls since there's only 14
    df = team_opponent.reset_index(drop = True).drop(columns = ['G', 'MP']).dropna()  
    
    ## Create a column to say whether or not the team made the playoffs
    df['playoffs'] = df.Team.str.contains('\*', regex = True).astype('int') 
    
    ## Format string for team names
    df['Team'] = df.Team.str.strip().str.replace(' ','_', regex = True)
    
    df['Team'] = df.Team.str.lower().str.replace('\*', '', regex = True)
    
    ## Split by whether or not the teams made the playoffs 
    all_teams = Normalize_Team_Data(df)  
    playoff_teams = all_teams[all_teams.playoffs == 1] 
    not_playoffs = all_teams[all_teams.playoffs == 0] 
    
    
    return {'all_teams' : all_teams, 
            'playoff_teams' : playoff_teams, 
            'not_playoffs' : not_playoffs}



## Given the formatted team data, look at the differences in stats based on playoff outcomes 
def TeamAverages(team_stats): 
    
    mean_fig = go.Figure() 
    
    ## For each set of team stats (playoffs, no playoffs, all_teams) 
    ## take the mean for each stat, and add them together in a bar graph
    for name in team_stats.keys(): 
        means = pd.DataFrame(team_stats[name].iloc[:,2:-1].mean()).transpose()
        mean_fig.add_trace(go.Bar(x = means.columns, y = means.iloc[0,:], name = name))
    
    mean_fig.update_layout(title_text = 'Differences in Stats Based on Playoff Outcome')
    
    ## We would also like to look at 
    
    return mean_fig


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





def ConvertCols(x): 
    
    ## Function to determine the winning percentage for each matchup 
    
    col = np.array(x.fillna('100000-11').str.split('-').to_list())

    col = col.astype(np.float64)
 
    wins = col.T[0] 
    losses = col.T[1] 
    return wins / np.add(wins, losses) 





def FormatStandings(standings_df): 
    
    ## Quick function to remove formatting for team names 
    FormatTeamNames = lambda df_team: df_team.str.strip().str.replace(' ', '_', regex = True).str.lower().str.replace(
                '\*', '', regex = True)
    
    ## Remove division statistics. We aren't going to be looking at them
    standings_df.columns = standings_df.columns.str.lower() 
    f_standings = standings_df[~standings_df.th.str.contains('Division', na = False)].copy() 
    
    ## Remove columns we don't need in this case - add year to index 
    f_standings.index = f_standings.year_y
    f_standings.drop(columns = ['year_x','year_y', 'th', 
                                        'w', 'l', 'w/l%', 'gb', 'ps/g', 
                                        'pa/g', 'srs', 'playoffs'], inplace = True)
    f_standings.reset_index(inplace = True) 
    f_standings.rename(columns = {'year_y' : 'year'}, inplace = True)
    
    ## Format team names. Drop null values
    f_standings.team = FormatTeamNames(f_standings.team)
    
    f_standings.dropna(how = 'all', axis = 1, inplace = True) 
    
    ## Determine winning percentage for each team matchup 
    f_standings = pd.concat([f_standings.iloc[:,0:2], 
                            f_standings.iloc[:,2:].transform(ConvertCols)], 
                            axis = 1)
    f_standings = f_standings.replace(f_standings.iloc[0,2],np.NaN)

    


    return f_standings.dropna(how = 'all')



def TeamWin(x): 
    if x < 0.4: 
        a = 0
    elif x <= 0.6: 
        a = 1
    else: 
        a = 2
    return a

def MatchColNames(df, year): 
    
    ## We want to match the column names with the team names in the team column 
    ## so to easily compare the teams and opponents in their matchups
    
    df_year = df[df.year == year].copy() 
    df_year = df_year.dropna(how = 'all', axis = 1) 
    
    names = [] 
    
    for col in df.columns: 
        try: 
            names.append(df_year[df_year[col].isna()].team.to_list()[0]) 
        except: 
            pass
        
        
    df_year.columns = ['year', 'team'] + names 
    return df_year.reset_index(drop = True) 


def GameOutcomes(df_year, year):
    
    ## Join together the team, opponent, and the outcomes in one single dataframe for each year
    
    game_outcomes = pd.DataFrame(columns = ['team', 'opponent', 'year', 'team_win']) 
    
    df_year.index = df_year.team 
    df_year.drop(columns = 'team', inplace = True) 
    team_names = df_year.index.unique() 
    for first_name in team_names: 
        for second_name in team_names: 
            team_win = df_year.unstack()[first_name][second_name] 
            game_outcomes = game_outcomes.append({
                'team': first_name, 
                'opponent': second_name, 
                'year': str(int(year)), 'team_win': team_win
            }, ignore_index = True)
        
    return game_outcomes

def ConcatOutcomes(standings): 
    
    ## For each year we are going to create a dataframe of matchups and their outomes 
    ## then we concat them together
    
    concat_games = pd.DataFrame(columns = ['team', 'opponent', 'year', 'team_win'])
    
    years = standings.year.unique() 
    for year in years[:-1]: 
        df_year = MatchColNames(standings, year) 
        games = GameOutcomes(df_year, year) 
        
        concat_games = pd.concat([concat_games, games], axis = 0) 
        
    concat_games.dropna(inplace = True) 
    
    ## Drop duplicate matchups 
    #concat_games['year'] = concat_games.year.astype(str) 
    concat_games['matchup'] = concat_games.apply(lambda x: sorted(x[['team', 'opponent', 'year']]), axis = 1) 
    concat_games = concat_games.groupby(concat_games.matchup.apply(tuple, 1)).first().reset_index(drop = True) 
    concat_games.drop(columns = ['matchup'], inplace = True) 
    concat_games['year'] = concat_games.year.astype(int) 
    
    ## Categorize the game outcomes just to make the final model easier 
    concat_games['team_win'] = concat_games.team_win.transform(TeamWin) 
    
        
    return concat_games 


def GameStats(all_team_stats, outcomes): 
    ## merge stats with the matchups 
    game_outcomes = outcomes.merge(
        all_team_stats,
        left_on = ['team', 'year'], 
        right_on = ['team', 'year']
    )
    game_outcomes = game_outcomes.merge(
        all_team_stats, 
        left_on = ['opponent', 'year'], 
        right_on = ['team', 'year']
    )
    game_outcomes.drop(columns = ['team_y'], inplace = True) 
    game_outcomes.rename(columns = {'team_x': 'team'}, inplace = True)
    
    ## We don't need the percentages since we have the makes and attempts 
    game_outcomes.drop(columns = [col for col in game_outcomes.columns if '%' in col], inplace = True)
    
    game_outcomes = game_outcomes.sample(1000) 
    
    ## Split into x and y so we can train the model
    x = game_outcomes.drop(columns = ['team', 'opponent', 'year', 'team_win']) 
    y = game_outcomes.team_win
    return game_outcomes, x, y