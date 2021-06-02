import pandas as pd 
from selenium import webdriver

class ParseURLS: 
    def __init__(self, year): 
        self.bball_url = 'https://www.basketball-reference.com/leagues/' 
        self.year = year
        self.hoopshype_salary_url = 'https://hoopshype.com/salaries/players/'

    def ToDF(self, text, headers, th = None, hoopshype = False): 
        ## Add the year column to the dataframe 
        
        df = pd.DataFrame(text, columns = headers) 
        df['year'] = self.year 
        if hoopshype is True: 
            return df
        elif th is not None: 
            df['Th'] = th

        return df
    
    def CleanURL(self, url, xpath): 
        '''find the stats given the url and the xpath of the
        table container'''
        
        browser = webdriver.Safari() 
        browser.get(url) 
        
        team_stats = browser.find_element_by_xpath(xpath) 
        
        
        
        
        ## Find all the headers 
        headers = team_stats.find_elements_by_tag_name('thead')[0] 
        headers = headers.text.split() 
        
        ## Find all the text 
        body = team_stats.find_elements_by_tag_name('tbody')[0]
        rows = body.find_elements_by_tag_name('tr') 
        
        ##Fill out all the rows 
        text = [] 
        th = []
        for row in rows: 
            elements = row.find_elements_by_tag_name('td') 
            th_element = row.find_element_by_tag_name('th')
            th.append(th_element.text)
            element_text = [] 
            for element in elements: 
                element_text.append(element.text) 
                
            text.append(element_text) 
        
        browser.quit() 

        return text, headers, th
    


    def Data(self, url, xpath, conference = False):


        text, headers, th = self.CleanURL(url, xpath)
        if conference is True: 
            df = self.ToDF(text, headers[2:], th) 
            df['playoffs'] = df.Th.apply(lambda x: 1 if '*' in x else 0) 
            df.Th = df.Th.str.replace('*','')
            return df
        
        df = self.ToDF(text, headers[1:]) 
        return df    
        
    def HoopsHype(self, url, xpath): 
        



    
        for j in range(10):
            try:
                
                browser = webdriver.Safari()
                browser.get(url) 
                browser.implicitly_wait(5)
                team_stats = browser.find_element_by_xpath(xpath) 
              
                    
            except: 
                print('{} hoops hype not found {}'.format(j, self.year))
                print('trying again') 
                browser.quit()
                continue
            if j == 9: 
                print('hoops hype xpath not found {}. \nReturning "None"'.format(self.year))
                return None
            break
            

        headers = team_stats.find_element_by_tag_name('thead') 
        headers = headers.text.split() 

        body = team_stats.find_element_by_tag_name('tbody') 

        rows = body.find_elements_by_tag_name('tr') 

        content = [] 
        for row in rows: 
            row_text = row.text.split()[1:] 

            while len(row_text) > 3: 
                row_text[0:2] = [' '.join(row_text[0:2])]
            
            content.append(row_text) 
        browser.quit() 

        headers = ['Player', 'Salary', 'adjusted_salary']
        player_salary = self.ToDF(
            content, headers, hoopshype=True
        )

        

        return player_salary

    def FormatPlayerNames(self,players_series):
        new_players_series = players_series.str.replace(
            ' II', '').replace(
                ' III', '').replace(
                    ' Jr', '').replace(
                        '.', '')
        new_players_series = new_players_series.str.strip()
        new_players_series = new_players_series.str.lower()

        return players_series