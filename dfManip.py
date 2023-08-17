# Funcaozinha para obter dos dados do Ministerio da Saude somente aquilo que julguei relevante para o TCC
import pandas as pd

def montarCSV(path):
    """
    Funcao que copia somente as colunas relevantes para um novo csv
    :param path: caminho do csv de entrada
    """
    """encoding='unicode_escape'"""
    df = pd.read_csv(path) # a mudanca de encoder foi necessaria para chik21
    df0 = df[[df.columns[0],'TP_NOT','ID_AGRAVO','DT_NOTIFIC','SEM_NOT','NU_ANO','SG_UF_NOT','SG_UF','EVOLUCAO','DT_OBITO']].copy()
    out = './' + path[-12:]
    df0.to_csv(path_or_buf=out)


# Abaixo, esquema pra plotar os csv do trends
df0 = pd.read_csv('./teste.csv')
df1 = pd.read_csv('./AC.csv')

a=3
pd.options.plotting.backend = "plotly"
fig0 = df0[df0.columns[1:]].plot()
fig0.update_layout(
    title_text='Search volume over time',
    legend_title_text='Search terms'
)
fig0.show()

fig1 = df1[df1.columns[1:]].plot()
fig1.update_layout(
    title_text='Search volume over time',
    legend_title_text='Search terms'
)
fig1.show()