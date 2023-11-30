# Umas funcao de manipular os dataframe sl
import pandas as pd
import os

# Codigos das unidades federativas conforme ta nas tabela do governo
UFCodes = {
    11 : 'RO', 12 : 'AC', 13 : 'AM', 14 : 'RR', 15 : 'PA', 16 : 'AP', 17 : 'TO', 
    21 : 'MA', 22 : 'PI', 23 : 'CE', 24 : 'RN', 25 : 'PB', 26 : 'PE', 27 : 'AL', 28 : 'SE', 29 : 'BA',
    31 : 'MG', 32 : 'ES', 33 : 'RJ', 35 : 'SP', 
    41 : 'PR', 42 : 'SC', 43 : 'RS',
    50 : 'MS', 51 : 'MT', 52 : 'GO', 53 : 'DF'
}

def fixCSV(path):
    """
    Funcao que copia somente as colunas relevantes para um novo csv
    :param path: caminho do csv de entrada
    """
    """encoding='unicode_escape'"""
    df = pd.read_csv(path) # a mudanca de encoder foi necessaria para chik21
    df0 = df[[df.columns[0],'TP_NOT','ID_AGRAVO','DT_NOTIFIC','SEM_NOT','NU_ANO','SG_UF_NOT','SG_UF','EVOLUCAO','DT_OBITO']].copy()
    out = './' + path[-12:] # Nomeia o arquivo
    df0.to_csv(path_or_buf=out)

def plotTrendsGraph(path):
    '''
    Funcao pra plotar grafico no estilo do trends
    :param path: caminho do csv de entrada
    '''
    df = pd.read_csv(path)

    pd.options.plotting.backend = "plotly"
    fig = df[df.columns[1:]].plot()
    fig.update_layout(
        title_text=path,
        legend_title_text='Search terms',
    )
    fig.show()

def plotAllTrends():
    '''
    Pra iterar em um diretorio de csvs do trends e plotar um por um
    '''
    directory = './aa/2010/' # Mudar aqui o caminho
    for file in os.scandir(directory):
        plotTrendsGraph(file.path)


# plotTrendsGraph('./trends_bib/2015/Rio Grande do Sul2015.csv')

df = pd.read_csv('./csv_resumidos/dengue15.csv')
a = 3
