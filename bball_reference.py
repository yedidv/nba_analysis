#%%
from numpy.core.numeric import NaN
import pandas as pd 
from selenium import webdriver
import numpy as np 
from BBallParent import ParseURLS



class BBallScrape(ParseURLS): 

    import pandas as pd 
    from selenium import webdriver
    import numpy as np 
    from BBallParent import ParseURLS
   
    def __init__(self, year): 
       super().__init__(year)
       
    
    def TeamStats(self): 
        ## Create a dataframe to bring in 
        ## the team data
        
        ## Read in the team url given the year
        url = self.bball_url + 'NBA_{}.html'.format(
            self.year)
        team_xpath = '//*[@id="team-stats-per_game"]'
        opponent_xpath = '//*[@id="opponent-stats-per_game"]'

        team_stats = super().Data(
            url,
            team_xpath)
        opponent_stats = super().Data(
            url, 
            opponent_xpath) 



        return team_stats, opponent_stats


    
    
    def Standings(self): 
        ## See how many wins each team has against each other
        url = self.bball_url + 'NBA_{}_standings.html'.format(self.year)
        standings_xpath = '//*[@id="team_vs_team"]'
        table_name = 'confs' if self.year >= 2016 else 'divs'
        eastern_conference_xpath = '//*[@id="{}_standings_E"]'.format(
            table_name)
        western_conference_xpath = '//*[@id="{}_standings_W"]'.format(
            table_name) 
            
        




        east = super().Data(
            url, 
            eastern_conference_xpath,
             conference = True) 
        west = super().Data(
            url, 
            western_conference_xpath, 
            conference = True) 
        standings = super().Data(
            url, 
            standings_xpath) 

        records = east.append(west) 

        standings = standings.merge(
            records, 
            left_on = 'Team', right_on = 'Th', 
            how = 'outer'
        )

        return standings
    
    def PlayerStats(self): 
        url = self.bball_url + 'NBA_{}_per_game.html'.format(
            self.year
        )
        players_xpath = '//*[@id="per_game_stats"]'

        player_stats = super().Data(
            url, players_xpath
        )


        
        
        
        player_stats['Player'] = super().FormatPlayerNames(player_stats['Player']) 





        return player_stats

    def PlayerSalary(self, current_year = False):
        if current_year is True: 
            url = self.hoopshype_salary_url
        else: 
            url = self.hoopshype_salary_url + '{}-{}/'.format(
                self.year - 1, self.year
            )
        
        salary_xpath = '//*[@id="content-container"]/div/div[3]/div[2]/table'



        player_salary = super().HoopsHype(
            url, salary_xpath
        )

        player_salary['Player'] = super().FormatPlayerNames(player_salary['Player']) 

        return player_salary
    
    def ReturnAllData(self): 

        self.player_salary = self.PlayerSalary(
            current_year = True if self.year is 2021 else False
        )

        self.player_stats = self.PlayerStats()
        
        self.standings = self.Standings() 
        self.team_stats, self.opponent_stats = self.TeamStats() 
        self.salary_cap = self.SalaryCap()

        self.all_stats = {
            'player_salary': self.player_salary, 
            'player_stats': self.player_stats, 
            'standings': self.standings, 
            'team_stats': self.team_stats, 
            'opponent_stats': self.opponent_stats, 
            'salary_cap': self.salary_cap
        }

        return self.all_stats
    
    def SalaryCap(self): 

        url = 'https://www.basketball-reference.com/contracts/salary-cap-history.html'


        browser = webdriver.Safari() 
        browser.get(url) 
        browser.implicitly_wait(5) 
        try: 
            head_text = True
            head = browser.find_element_by_xpath(
            '//*[@id="salary_cap_history"]/thead'
            )

            body = browser.find_element_by_xpath(
                    '//*[@id="salary_cap_history"]/tbody'
            )
            body = body.find_elements_by_tag_name('tr')



        except: 
            head_text = False
            head = browser.find_element_by_xpath(
            '//*[@id="salary_cap_history"]/tbody/tr[1]'
            )
            body = browser.find_element_by_xpath(
            '//*[@id="salary_cap_history"]/tbody'
            )
            body = body.find_elements_by_tag_name('tr')[1:]
        print(body[0].text)

        table_body = []
        for row in body: 
            row_text = row.text.split('$')
            table_body.append(row_text) 

        head = [x.strip() for x in head.text.split('\n')]
        head = [x for x in head if x]
        print(head_text) 

        self.salary_cap = pd.DataFrame(table_body, columns = head) 


        browser.quit()

        return self.salary_cap

 








