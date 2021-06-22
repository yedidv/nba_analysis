from selenium import webdriver 
import pandas as pd 
import numpy as np 

class BBallReferenceData: 
    def __init__(self, year): 
        self.year = year 
        bball_url = 'https://www.basketball-reference.com/leagues/'
        self.bball_url = bball_url + 'NBA_{}.html'.format(self.year)
        self.standings_url = bball_url + 'NBA_{}_standings.html'.format(self.year) 
        
    def CleanTable(self, table_info): 
        ## Given the table xpaths, scrape the data
        ## and place it into a dataframe 
        
        ## The browser URL should already be open 
        ## Find the table headers 
        headers = table_info.find_elements_by_tag_name('thead')[0] 
        headers = headers.text.split() 
        headers = [header.lower() for header in headers]


        ## Find all the rest of the table elements 
        body = table_info.find_elements_by_tag_name('tbody')[0] 
        rows = body.find_elements_by_tag_name('tr') 

        ## Fill out all the rows 
        text = [] 
        th = [] 
        for row in rows: 
            elements = row.find_elements_by_tag_name('td') 
            try: 
                th_element = row.find_element_by_tag_name('th') 
                th.append(th_element.text) 
            except: 
                pass
            element_text = [] 
            for element in elements: 
                element_text.append(element.text) 
            text.append(element_text) 
        
        
        return {'th': th, 'text': text, 'headers': headers }

    def StandingsToDF(self, raw_data): 
        

        ## Now put all the standings data in a dataframe 
        df = pd.DataFrame() 
        df['team'] = raw_data['th']
        df['year'] = self.year 
        team_data = pd.DataFrame(raw_data['text'], columns = raw_data['headers'][2:]) 
        df = pd.concat([df, team_data], axis = 1) 

        df = df[~df.team.str.contains('Division')]
        df.team = df.team.str.replace('\*', '', regex = True) 


        return df 
    
    def StatsToDF(self, raw_data): 
        ## put all the stats data in a dataframe 
        df = pd.DataFrame() 
        df['year'] = self.year 
        df = pd.DataFrame(raw_data['text'], columns = raw_data['headers'][1:])
        df.insert(loc = 1, column = 'year', value = self.year) 
        df.dropna(inplace = True)
        df.insert(
            loc = 2, 
            column = 'playoff', 
            value = df.team.str.contains('\*', regex = True).astype('int'))
        df.team = df.team.str.replace('\*', '', regex = True) 
        return df
    
    def PlayoffsToDF(self, raw_data): 
        ## Take data from playoff outcomes, 
        ## clean it and put it in a dataframe 
        df = pd.DataFrame(raw_data['text']) 
        df.dropna(subset = [1], inplace = True) ## drop na for column containing score 
        df = df[df[1].str.contains('\n')][1].str.split('\n') ## show rows containing score
        df = pd.DataFrame(df.to_list(), ## convert score to dataframe of its own
                         columns = ['winner', 'over', 'loser', 'score', 'empty'])
        df.drop(columns = ['over', 'empty'], inplace = True) ## drop columns we don't need
        df.score = df.score.str.replace('(', '', regex = True) 
        df.score = df.score.str.replace(')', '', regex = True) 
        df.score = pd.DataFrame(df.score.str.split('-').to_list())[1] ##only keep losing score
        df.insert(loc = 2, column = 'year', value = self.year) 
        return df
    
    def H2HToDF(self, raw_data): 
        ## Clean up the data for regular season matchups, 
        ## and put them in a dataframe 
        df = pd.DataFrame(raw_data['text'], columns = raw_data['headers'][1:]) 
        ## put the team name to the index
        df.index = df.team 
        df.drop(columns = 'team', inplace = True) 
        
        ## Rename the columns to match the full team names 
        names = [] 
        for col in df.columns: 
            names.append(df[df[col] == df.iloc[1,1]].index.to_list()[0])
        df.columns = names 
        
        ## unstack the standings to look at head to head matchups in one column
        game_outcomes = pd.DataFrame(columns = ['team', 'opponent', 'year', 'record']) 
        
        for team1 in df.index.unique().to_list(): 
            for team2 in df.index.unique().to_list(): 
                try: 
                    team_win = df.unstack()[team1][team2] 
                    if team_win != df.iloc[1,1]: 
                        game_outcomes = game_outcomes.append(
                            {'team' : team1, 
                            'opponent' : team2, 
                            'year' : self.year, 
                            'record' : team_win}, ignore_index = True
                        )
                except: pass

        return game_outcomes
    
    def RemoveTeamFormatting(self, df): 
        ## Quick function to remove teeam formatting 
        ## from the dataframe 
        
        df = df.str.lower().str.strip() 
        df = df.str.replace(' ', '_', regex = True)
        return df 
    
    def FindYear(self): 
        ## For each year we want to look at the URL, scrape info from the tables 
        ## to find standings data, playoff data, and per game stats
        
        browser = webdriver.Safari() 
        browser.get(self.bball_url) 
        
        # The NBA removed Divisions after 2016
        table_name = 'confs' if self.year >= 2016 else 'divs'
        
        ## Find the xpath for each of the conferences given the year 
        
        ## Start with east
        east_xpath = '//*[@id="{}_standings_E"]'.format(table_name)
        east_standings = browser.find_element_by_xpath(east_xpath)
        east_standings_data = self.CleanTable(east_standings) 
        east_standings = self.StandingsToDF(east_standings_data) 
        
        ## West standings 
        west_xpath = '//*[@id="{}_standings_W"]'.format(table_name)
        west_standings = browser.find_element_by_xpath(west_xpath)
        west_standings_data = self.CleanTable(west_standings) 
        west_standings = self.StandingsToDF(west_standings_data) 
        
        standings = pd.concat([east_standings, west_standings], axis = 0)
        
        ## Playoff Outcomes 
        playoff_xpath = '//*[@id="all_playoffs"]'
        playoff_xpath = browser.find_element_by_xpath(playoff_xpath) 
        playoffs_data = self.CleanTable(playoff_xpath) 
        playoffs = self.PlayoffsToDF(playoffs_data) 
        
        ## Per Game Stats
        stats_xpath = '//*[@id="per_game-team"]'
        stats = browser.find_element_by_xpath(stats_xpath)
        stats_data = self.CleanTable(stats)
        stats = self.StatsToDF(stats_data) 
        
        ## Regular Season Head to head matchup outcomes
        browser.get(self.standings_url)
        h2h_xpath = '//*[@id="team_vs_team"]'
        h2h = browser.find_element_by_xpath(h2h_xpath) 
        h2h_data = self.CleanTable(h2h) 
        h2h = self.H2HToDF(h2h_data) 

        
        
        
        ## Quit the browser
        browser.quit()
        
        ## Format the team names for 
        ## each of the dataframes
        standings.team = self.RemoveTeamFormatting(
            standings.team
        ) 
        playoffs.winner = self.RemoveTeamFormatting(
            playoffs.winner
        ) 
        playoffs.loser = self.RemoveTeamFormatting(
            playoffs.loser
        ) 
        stats.team = self.RemoveTeamFormatting(
            stats.team
        )
        h2h.team = self.RemoveTeamFormatting(
            h2h.team
        ) 
        h2h.opponent = self.RemoveTeamFormatting(
            h2h.opponent
        ) 
        
        
        
        
        return {'standings': standings,
                'playoffs': playoffs,
                'stats': stats, 
                'h2h': h2h}
    
    
    
    
    
    
    
