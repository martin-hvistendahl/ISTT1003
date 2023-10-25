import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

def cleanData():
    df = pd.read_csv("lego.population.csv", sep = ",", encoding = "latin1")

    # fjerner forklaringsvariabler vi ikke trenger
    df2 = df[['Set_Name', 'Theme', 'Pieces', 'Price', 'Pages', 'Minifigures', 'Unique_Pieces']]

    # fjerner observasjoner med manglende datapunkter
    df2 = df2.dropna()

    # gjør themes om til string og fjern alle tegn vi ikke vil ha med
    df2['Theme'] = df2['Theme'].astype(str)
    df2['Theme'] = df2['Theme'].str.replace(r'[^a-zA-Z0-9\s-]', '', regex = True)

    # fjerner dollartegn og trademark-tegn fra datasettet
    df2['Price'] = df2['Price'].str.replace('\$', '', regex = True)

    # og gjør så prisen om til float
    df2['Price'] = df2['Price'].astype(float)

    gutt_theme=["NINJAGO", "Star Wars", "Marvel", "Batman","Speed Champions", "Hidden Side","Jurassic World", "Overwatch", "Spider-Man", "DC", "Monkie Kid", "Powered UP"]
    jente_theme=["Disney","Friends", "Unikitty", "LEGO Frozen 2", "Trolls World Tour", "Powerpuff Girls"]

    #if df2['Theme'] = gutt_theme sett gutt, elif df2['Theme'] = jente else neutral
    df2['gender'] = np.where(df2['Theme'].isin(gutt_theme), 'boy', np.where(df2['Theme'].isin(jente_theme), 'girl', 'neutral'))

    #create a new .csv file with Pices, Price, Pages, Unique_Pieces and kjønn
    df2[["Pieces", "Price", "Pages", "Unique_Pieces", "gender"]].to_csv('lego.population2.csv', index=False)


def createModell():
    df2 = pd.read_csv("lego.population2.csv", sep=",", encoding="latin1")

    categories = ['boy', 'girl', 'neutral']
    
    for category in categories:
        df_subset = df2[df2['gender'] == category]
        
        plt.figure()  # Create a new figure for each category
        
        if len(df_subset) > 1:  # Check if there are at least two data points for regression
            # Enkel lineær regresjon
            formel = 'Price ~ Pieces'
            modell = smf.ols(formel, data=df_subset)
            resultat = modell.fit()
            
            print(f"Summary for {category}:\n", resultat.summary())
            
            slope = resultat.params['Pieces']
            intercept = resultat.params['Intercept']
            
            regression_x = np.array(df_subset['Pieces'])
            regression_y = slope * regression_x + intercept
            
            plt.scatter(df_subset['Pieces'], df_subset['Price'], label='Data Points')
            plt.plot(regression_x, regression_y, color='red', label='Regression Line')
            plt.text(min(df_subset['Pieces']), max(df_subset['Price']), f'y = {slope:.2f}x + {intercept:.2f}', color='red')
        else:
            print(f"Not enough data points for {category} category to fit a regression model.")
            plt.text(0.5, 0.5, 'Not enough data points', horizontalalignment='center', verticalalignment='center')
        
        plt.xlabel('Antall brikker')
        plt.ylabel('Pris [$]')
        plt.title(f'Kryssplott med regresjonslinje (enkel LR) for {category}')
        plt.legend()
        plt.grid()
        plt.show()

createModell()