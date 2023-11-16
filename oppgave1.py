import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split    
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm


df2 = pd.read_csv("lego.population2.csv", sep=",", encoding="latin1")
df = pd.read_csv("lego.population.csv", sep=",", encoding="latin1")

def cleanData():
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

    gutt_theme=["NINJAGO", "Star Wars", "Marvel", "Batman","Speed Champions", "Hidden Side","Jurassic World", "Overwatch", "Spider-Man", "DC", "Monkie Kid", "Powered UP", "Technic"]
    jente_theme=["Disney","Friends", "Unikitty", "LEGO Frozen 2", "Trolls World Tour", "Powerpuff Girls", "DOTS"]

    #if df2['Theme'] = gutt_theme sett gutt, elif df2['Theme'] = jente else neutral
    df2['gender'] = np.where(df2['Theme'].isin(gutt_theme), 'boy', np.where(df2['Theme'].isin(jente_theme), 'girl', 'neutral'))

    #create a new .csv file with Pices, Price, Pages, Unique_Pieces and kjønn
    df2[["Pieces", "Price", "Pages", "Unique_Pieces", "gender"]].to_csv('lego.population2.csv', index=False)
    print(df2.loc[df2['Price'].idxmax()])

def dataInformation():
    #lage en stolpediagram med antall og pris for alle gruppene sammen
    plt.hist(df2['Price'], bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Pris i dollar [$]')
    plt.ylabel('Antall sett')
    plt.gca().set_aspect(1)
    plt.show()
    
    plt.scatter(df2['Pieces'], df2['Price'])
    plt.xlabel('Antall brikker')
    plt.ylabel('Pris i dollar [$]')
    plt.gca().set_aspect(5)
    plt.show()
    
    #find min max mean, mediam
    print("min",df2['Price'].min())
    print("max",df2['Price'].max())
    print("gjennomsnitt",df2['Price'].mean())
    print("median",df2['Price'].median())
    
def metode1():
    formel = 'Price ~ Pieces'
    model = smf.ols('Price ~ Pieces', data=df2).fit()
    modell = smf.ols(formel, data = df2)
    # resultat = modell.fit()
    
    print(model.summary())

    figure, axis = plt.subplots(1, 2, figsize = (15, 5))
    sns.scatterplot(x = model.fittedvalues, y = model.resid, ax = axis[0])
    axis[0].set_ylabel("Residual")
    axis[0].set_xlabel("Predikert verdi")

    sm.qqplot(model.resid, line = '45', fit = True, ax = axis[1])
    axis[1].set_ylabel("Kvantiler i residualene")
    axis[1].set_xlabel("Kvantiler i normalfordelingen")
    plt.show()   
    
    slope = model.params['Pieces']
    intercept = model.params['Intercept']
    regression_function = f'Price = {slope:.2f} * x + {intercept:.2f}'

    df2['predicted_price'] = model.predict(df2['Pieces'])

    plt.scatter(df2['Pieces'], df2['Price'], label='Reelle priser')
    plt.plot(df2['Pieces'], df2['predicted_price'], color='red', label='Regresjonslinje')

    plt.xlabel('Antall brikker')
    plt.ylabel('Pris [$]')
    plt.title(f'Regresjonsanalyse: Pris beskrevet av antall brikker\n{regression_function}')
    
    max_pieces = int(np.ceil(max(df2['Pieces']) / 1000) * 1000)
    max_price = int(np.ceil(max(df2['Price']) / 100) * 100)
    
    plt.xticks(np.arange(0, max_pieces + 1, 1000))
    plt.yticks(np.arange(0, max_price + 1, 100))

    plt.legend()
    plt.grid(True)
    plt.show()    

def metode2():

    X = df2[['Pieces', 'Pages']]
    y = df2['Price']
    
    if len(X) > 1:
        model = LinearRegression().fit(X, y)
        
        x_surf, y_surf = np.meshgrid(np.linspace(X['Pieces'].min(), X['Pieces'].max(), 100), 
                                     np.linspace(X['Pages'].min(), X['Pages'].max(), 100))
        z_surf = model.intercept_ + model.coef_[0] * x_surf + model.coef_[1] * y_surf
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(X['Pieces'], X['Pages'], y, color='b', label='Data Points')
        ax.plot_surface(x_surf, y_surf, z_surf, color='r', alpha=0.3)
        
        ax.set_xlabel('Antall brikker')
        ax.set_ylabel('Sider i bruksanvisning')
        ax.set_zlabel('Pris [$]')
        ax.set_title('3D-plot av Lego-sett: Pris beskrevet av antall brikker og sider')
        
        # Creating a custom legend
        scatter_proxy = plt.Line2D([0],[0], linestyle="none", c='b', marker = 'o')
        surface_proxy = plt.Rectangle((0,0),1,1,fc="r", alpha=0.3)
        equation = f"Price = {model.intercept_:.2f} + ({model.coef_[0]:.2f} * Pieces) + ({model.coef_[1]:.2f} * Pages)"
        ax.text2D(0.05, 0.95, equation, transform=ax.transAxes)

        plt.show()
    else:
        print("Not enough data points to fit a regression model.")
 
def metode3a():

    categories = ['boy', 'girl', 'neutral']

    max_pieces = int(np.ceil(max(df2['Pieces']) / 1000) * 1000)
    max_price = int(np.ceil(max(df2['Price']) / 100) * 100)

    for category in categories:
        df_subset = df2[df2['gender'] == category]
        
        plt.figure()  # Create a new figure for each category
        
        title = f'Kryssplott med regresjonslinje (enkel LR) for {category}'
        
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
            title += f'\nRegression Formula: y = {slope:.2f}x + {intercept:.2f}'
        else:
            print(f"Not enough data points for {category} category to fit a regression model.")
            plt.text(0.5, 0.5, 'Not enough data points', horizontalalignment='center', verticalalignment='center')
        
        plt.xlabel('Antall brikker')
        plt.ylabel('Pris [$]')
        plt.title(title)
        plt.xticks(np.arange(0, max_pieces + 1, 1000))
        plt.yticks(np.arange(0, max_price + 1, 100))
        plt.legend()
        plt.grid()
        plt.show()   

def metode3b():
    modell3_mlr = smf.ols('Price ~ Pieces + C(gender)', data = df2)
    resultat = modell3_mlr.fit()

    print(resultat.summary())

    # Definerer kategorier og farger
    categories = ['boy', 'girl', 'neutral']
    colors = {'boy': 'dodgerblue', 'girl': 'black', 'neutral': 'green'}

    # Plott for hvert kjønn
    for category in categories:
        # Filtrerer data for hver kategori
        subset = df2[df2['gender'] == category]

        # Beregner forventet pris basert på modellen
        predicted_price = resultat.predict(subset)

        # Plotter scatter plot og regresjonslinje
        plt.scatter(subset['Pieces'], subset['Price'], color=colors[category], label=category)
        plt.plot(subset['Pieces'], predicted_price, color=colors[category])

    plt.xlabel('Antall brikker')
    plt.ylabel('Pris')
    plt.title('Kryssplott med regresjonslinjer fra MLR-modell')
    plt.legend()
    plt.grid()
    plt.show()

def metode3c():
    # Fitte modellen med interaksjonstermen
    model = smf.ols('Price ~ Pieces * C(gender)', data=df2)
    results = model.fit()
    print(results.summary())
    
    # Kategorier og farger
    categories = ['boy', 'girl', 'neutral']
    colors = {'boy': 'dodgerblue', 'girl': 'magenta', 'neutral': 'green'}
    
    # Plotte de forskjellige linjene
    for category in categories:
        # Filtrer data for hver kategori
        subset = df2[df2['gender'] == category]
        
        # Beregne forventede verdier basert på modellen
        subset['predicted_price'] = results.predict(subset)
        
        # Sorter for plotting
        subset_sorted = subset.sort_values('Pieces')
        
        # Plotte linjen for hver kategori
        plt.plot(subset_sorted['Pieces'], subset_sorted['predicted_price'], 
                 color=colors[category], label=f'{category} line')
        
        # Plotte datapunktene for hver kategori
        plt.scatter(subset['Pieces'], subset['Price'], 
                    color=colors[category], alpha=0.5, label=f'{category} data')
    
    # Plottegenskaper
    plt.xlabel('Antall brikker')
    plt.ylabel('Pris [$]')
    plt.title('Kryssplott med regresjonslinjer for hver kjønnskategori')
    plt.legend()
    plt.grid(True)
    plt.show()

def metode4():
    # Opprette interaksjonstermer for Pieces og Pages med gender
    model_formula = 'Price ~ C(gender):Pieces + C(gender):Pages'

    # Fitte modellen
    model = smf.ols(model_formula, data=df2).fit()
    
    # Printe modellens sammendrag
    print(model.summary())

