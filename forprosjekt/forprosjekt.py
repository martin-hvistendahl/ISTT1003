import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = 'gruppens_datasett.xlsx'


def load_data(file_path):
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")
    else:
        return df
        
       
data=load_data(file_path)

def plotGraf(data):
    if data is None:
        return
    
    x = data['skostr']
    y = data['hoyde']

    b, a = np.polyfit(x, y, 1) 
    plt.scatter(x, y, color='blue', label='Data')
    plt.plot(x, a + b*x, color='red', label=f'Regresjonslinje: y = {a:.2f} + {b:.2f}x')
    plt.xlabel('Skostørrelse')
    plt.ylabel('Høyde')
    plt.title('Kryssplot og regresjonslinje')
    plt.legend(loc='best')

    pdf_path = 'regresjonslinje_plot.pdf'
    plt.savefig(pdf_path, format='pdf')
    plt.show()

if data is not None:
    plotGraf(data)
else:
    print("No data to plot")