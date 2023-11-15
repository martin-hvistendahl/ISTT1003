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

def numberOfObservations():
    # Number of observations for boys
    df_subset_boy = df2[df2['gender'] == 'boy']
    print(len(df_subset_boy), 'Observasjoner rettet mot gutter')

    # Number of observations for girls
    df_subset_girl = df2[df2['gender'] == 'girl']
    print(len(df_subset_girl), 'Observasjoner rettet mot jenter')

    # Number of observations for neutral
    df_subset_girl = df2[df2['gender'] == 'neutral']
    print(len(df_subset_girl), 'Kjønnsnøytrale observasjoner')

    # Number of unique themes
    unique_themes = df['Theme'].unique()
    print(len(unique_themes), 'Unike temaer')

    # Create a DataFrame from the unique themes and save it to a CSV file
    themes_df = pd.DataFrame({'Theme': unique_themes})
    themes_df.to_csv('lego.themes.csv', index=False)

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

#1 Pris beskrevet av antall brikker.
def regression_plot_price_pieces():
    df2 = pd.read_csv("lego.population2.csv", sep=",", encoding="latin1")

    # Enkel lineær regresjon
    model = smf.ols('Price ~ Pieces', data=df2).fit()
    print(model.summary())

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

# 2 Pris beskrevet av antall brikker og antall sider i bruksanvisningen.
def regression_plot_price_pieces_pages():

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
#3a Pris beskrevet av antall brikker for de forskjellige gruppene; gutt, jente og nøytral
def regression_plot_price_pieces_by_gender():

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
#3b Regresjonslinjene for de forskjellige gruppene inn i en modell.
def regression_plot_price_pieces_gender_combined():
    categories = ['boy', 'girl', 'neutral']
    colors = {'boy': 'dodgerblue', 'girl': 'magenta', 'neutral': 'green'}

    # Set up the plot
    plt.figure(figsize=(10, 6))

    # Loop through each category
    for category in categories:
        # Filter the data for the current category
        df_subset = df2[df2['gender'] == category]

        # Prepare the data for regression
        X = df_subset['Pieces']
        X = sm.add_constant(X)  # Adds a constant term to the predictor
        y = df_subset['Price']

        # Fit the model
        model = sm.OLS(y, X).fit()

        # Make predictions for the line
        x_vals = np.linspace(X['Pieces'].min(), X['Pieces'].max(), 100)
        y_vals = model.params['const'] + model.params['Pieces'] * x_vals

        # Plot the line
        plt.plot(x_vals, y_vals, label=f'{category.capitalize()} (y = {model.params["Pieces"]:.2f}x + {model.params["const"]:.2f})', color=colors[category])

        # Plot the data points
        plt.scatter(X['Pieces'], y, alpha=0.5, color=colors[category])

    # Customize the plot
    plt.xlabel('Pieces')
    plt.ylabel('Price')
    plt.title('Price vs. Pieces by Gender')
    plt.legend()
    plt.grid(True)
    plt.show()

#4 Pris beskrevet av antall brikker og antall sider i bruksanvisningen for de forskjellige gruppene;  gutt, jente og nøytral.
def regression_plot_price_pieces_pages_gender():
    df2 = pd.read_csv("lego.population2.csv", sep=",", encoding="latin1")

    categories = ['boy', 'girl', 'neutral']
    
    for category in categories:
        df_subset = df2[df2['gender'] == category]
        
        if len(df_subset) > 1:
            X = df_subset[['Pieces', 'Pages']]
            y = df_subset['Price']
            
            model = LinearRegression().fit(X, y)
            
            x_surf, y_surf = np.meshgrid(np.linspace(X['Pieces'].min(), X['Pieces'].max(), 100), np.linspace(X['Pages'].min(), X['Pages'].max(), 100))
            z_surf = model.intercept_ + model.coef_[0] * x_surf + model.coef_[1] * y_surf
            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            ax.scatter(X['Pieces'], X['Pages'], y, color='b', label='Data Points')
            ax.plot_surface(x_surf, y_surf, z_surf, color='r', alpha=0.3, label='Regression Surface')
            
            ax.set_xlabel('Antall brikker')
            ax.set_ylabel('Sider i bruksanvisning')
            ax.set_zlabel('Pris [$]')
            ax.set_title(f'3D-plot av Lego-sett ({category})')
            
            plt.show()
            
            model = smf.ols('Price ~ Pieces + Pages', data=df_subset).fit()
            print(model.summary()) 
           
        
        else:
            print(f"Not enough data points for {category} category to fit a regression model.")

regression_plot_price_pieces_pages_gender()
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
    
    formel = 'Price ~ Pieces'

    modell = smf.ols(formel, data = df2)
    resultat = modell.fit()
    
    print(resultat.summary())

    figure, axis = plt.subplots(1, 2, figsize = (15, 5))
    sns.scatterplot(x = resultat.fittedvalues, y = resultat.resid, ax = axis[0])
    axis[0].set_ylabel("Residual")
    axis[0].set_xlabel("Predikert verdi")

    sm.qqplot(resultat.resid, line = '45', fit = True, ax = axis[1])
    axis[1].set_ylabel("Kvantiler i residualene")
    axis[1].set_xlabel("Kvantiler i normalfordelingen")
    plt.show()
    
    
def reg_konf():   
    formel = 'Price ~ Pieces'

    modell = smf.ols(formel, data=df2)
    resultat = modell.fit()    
    slope = resultat.params['Pieces']
    intercept = resultat.params['Intercept']

    regression_x = np.array(df2['Pieces'])
    regression_y = slope * regression_x + intercept
    predictions = resultat.get_prediction(df2).summary_frame(alpha=0.05)  # 95% konfidensintervall

    # Plukker ut nedre og øvre grense for konfidensintervallene
    ci_lower = predictions['obs_ci_lower']
    ci_upper = predictions['obs_ci_upper']

    # Plotter datapunktene
    plt.scatter(df2['Pieces'], df2['Price'], label='Data Points')

    # Plotter regresjonslinjen
    plt.plot(regression_x, regression_y, color='red', label='Regression Line')

    # Plotter konfidensintervallene
    plt.fill_between(regression_x, ci_lower, ci_upper, color='blue', alpha=0.1, label='95% CI')

    # Plottegenskaper
    plt.xlabel('Antall brikker')
    plt.ylabel('Pris [$]')
    plt.title('Kryssplott med regresjonslinje og konfidensintervall')
    plt.legend()
    plt.grid(True)
    plt.show()  

def skjæringspunkt5CB():
    resultater = []
    categories = ['boy', 'girl', 'neutral']
    colors = {'boy': 'dodgerblue', 'girl': 'magenta', 'neutral': 'green'}
    #make the girl dots and line black
    for i, category in enumerate(categories):
        modell3 = smf.ols('Price ~ Pieces' , data = df2[df2['gender'].isin([category])])
        resultater.append(modell3.fit())
        
    for i, category in enumerate(categories):
        slope = resultater[i].params['Pieces']
        intercept = resultater[i].params['Intercept']

        regression_x = np.array(df2[df2['gender'].isin([category])]['Pieces'])
        regression_y = slope * regression_x + intercept

        # Plot scatter plot and regression line
        plt.scatter(df2[df2['gender'].isin([category])]['Pieces'], df2[df2['gender'].isin([category])]['Price'], color=colors[category])
        plt.plot(regression_x, regression_y, color=colors[category], label=category)
        
    plt.xlabel('Antall brikker')
    plt.ylabel('Pris')
    plt.title('Kryssplott med regresjonslinjer')
    plt.legend()
    plt.grid()
    plt.show()


    # multippel lineær regresjon
    modell3_mlr = smf.ols('Price ~ Pieces + gender' , data = df2)
    modell3_mlr.fit().summary()

    print(modell3_mlr.fit().summary())

def skjæringspunkt5CC():
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
# skjæringspunkt5CB()    
# skjæringspunkt5CC()