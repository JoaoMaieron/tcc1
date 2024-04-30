# Umas funcao de manipular os dataframe do sus
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Codigos das unidades federativas conforme ta nas tabela do governo
UFCodes = {
    11 : 'RO', 12 : 'AC', 13 : 'AM', 14 : 'RR', 15 : 'PA', 16 : 'AP', 17 : 'TO', 
    21 : 'MA', 22 : 'PI', 23 : 'CE', 24 : 'RN', 25 : 'PB', 26 : 'PE', 27 : 'AL', 28 : 'SE', 29 : 'BA',
    31 : 'MG', 32 : 'ES', 33 : 'RJ', 35 : 'SP', 
    41 : 'PR', 42 : 'SC', 43 : 'RS',
    50 : 'MS', 51 : 'MT', 52 : 'GO', 53 : 'DF'
}

years = [str(n) for n in range(2010,2023)]




df = pd.read_csv('../dados_versao_final/trends/RS.csv',index_col='Week',parse_dates=True)
plt.ioff()
fig, ax = plt.subplots(figsize=(10,6))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.plot(df.index,df['buscas'], label='buscas pelo termo')
# ax.set_ylabel('Casos')
ax.set_title('Trends ''dengue'' - RS')
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()
# plt.savefig(fname = uf+'.png',format='png')

for uf in UFCodes.values():
    df = pd.read_csv('../dados_versao_final/trends/'+uf+'.csv',index_col='Week',parse_dates=True)
    plt.ioff()
    fig, ax = plt.subplots(figsize=(10,6))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.plot(df.index,df['buscas'], label='buscas pelo termo')
    # ax.set_ylabel('Casos')
    ax.set_title('Trends ''dengue'' - '+uf)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fname = uf+'.png',format='png')


import fileinput
def removeGarbage():
    '''
    Removendo as duas primeiras linhas dos csv do trends
    '''
    for y in years:
        folder_path = 'C:\\Users\\joaom\\Desktop\\tcc\\dados\\dados_em_tratamento\\csv_trends_manual\\'+y+'\\'
        csv_files = [file for file in os.listdir(folder_path)]
        for file in csv_files:
            file_path = os.path.join(folder_path, file)
            
            # Open the file in place, with inplace=True
            with fileinput.FileInput(file_path, inplace=True) as current_file:
                # Skip the first two lines
                for i, line in enumerate(current_file):
                    if i < 2:
                        continue
                    print(line, end="")

def concatTrends():
    '''
    Concatenar os csv do trends
    '''
    for uf in UFCodes.values():
        dfs = []
        for y in years:
            df = pd.read_csv('./csv_trends_manual/'+y+'/'+uf+'.csv')
            dfs.append(df)
        result_df = pd.concat(dfs, ignore_index=True)
        result_df.to_csv(uf+'.csv',index=False)

def showDeBola():
    '''
    Pega só a coluna da dengue nos csv do trends
    '''
    for uf in UFCodes.values():
        df = pd.read_csv('./trends/'+uf+'.csv').iloc[:, [0,2]]
        # print(uf,' ',len(df))
        df = df.rename(columns={df.columns[1] : 'buscas'})
        df.to_csv(uf+'.csv',index=False)
showDeBola()
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

'''
Para contar os casos de cada UF:
df.groupby('SG_UF_NOT').nunique()

Para contar os óbitos de cada UF:
df.groupby('SG_UF_NOT')['DT_OBITO'].nunique()

Para contar total de óbitos:
df['DT_OBITO'].count()
'''
