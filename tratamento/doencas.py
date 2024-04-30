# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
UFCodes = {
    11 : 'RO', 12 : 'AC', 13 : 'AM', 14 : 'RR', 15 : 'PA', 16 : 'AP', 17 : 'TO', 
    21 : 'MA', 22 : 'PI', 23 : 'CE', 24 : 'RN', 25 : 'PB', 26 : 'PE', 27 : 'AL', 28 : 'SE', 29 : 'BA',
    31 : 'MG', 32 : 'ES', 33 : 'RJ', 35 : 'SP', 
    41 : 'PR', 42 : 'SC', 43 : 'RS',
    50 : 'MS', 51 : 'MT', 52 : 'GO', 53 : 'DF'
}


# UFCodes = {
#     32 : 'ES'
# }


years = [str(n) for n in range(2010,2023)]

dfPenis = pd.read_csv('./doencas/AL.csv')
semanas = dfPenis['semana'].values # ISSO AQUI É IMPORTANTISSIMO POIS VAI SER A INDEX DOS DATAFRAME

#==============================================================================
# PRE PROCESSAMENTO
#==============================================================================

# Pegando só as colunas relevantes
# INICIA COM OS CSV CRU DO SINAN
folder_path = 'C:\\Users\\joaom\\Desktop\\tcc\\dados\\dados_brutos\\dados_sinan\\'
csv_files = [file for file in os.listdir(folder_path)]
for file in csv_files:
    file_path = os.path.join(folder_path,file)
    df = pd.read_csv(file_path)
    if 'CLASSI_FIN' in df.columns.values:
        df = df[~df['CLASSI_FIN'].isin([5, 13])]
    df0 = df[['SEM_NOT','DT_NOTIFIC','SEM_PRI','SG_UF_NOT','DT_OBITO']].copy()
    df0.to_csv('../dados_brutos/a/'+file+'.csv',index=False)

# file = '../dados_em_tratamento/xablau/DENGBR18.csv.csv'

# PARA 2020 TAVA TUDO ERRADO O CSV ????
# O nome das colunas tava errado, aqui corrige
df = pd.read_csv('../dados_brutos/dados_sinan/DENGBR20.csv')
df = df[~df['CLASSI_FIN'].isin([5, 13])]
df0 = df[['DT_NOTIFIC','SEM_PRI','SG_UF_NOT','DT_OBITO']].copy()
df0 = df0[df0['SEM_PRI'] > 202000] # EU ESQUECI DE ARRUMAR AQUI, NAO CONTA PRA NGM
df0['SEM_PRI'] = df0['SEM_PRI'].astype(int)
df0 = df0.rename(columns={'SEM_PRI':'SEM_NOT'})
df0.to_csv(path_or_buf='../dados_brutos/dados_sinan/DENGBR20.csv',index=False)

# Concatenando todos em um gigantao
# folder_path = 'C:\\Users\\joaom\\Desktop\\tcc\\dados\\dados_brutos\\a\\'
folder_path = 'C:\\Users\\joaom\\Desktop\\tcc\\dados\\modelos_LSTM\\TESTE\\a\\'
csv_files = [file for file in os.listdir(folder_path)]
dfs = []    
for file in csv_files:
    file_path = os.path.join(folder_path,file)
    df = pd.read_csv(file_path)
    dfs.append(df)
result_df = pd.concat(dfs, ignore_index=True)
result_df.to_csv('metricas.csv',index=False)

# Ha uma inconsistencia no codigo das semanas, aqui corrige
df = pd.read_csv('../dados_brutos/DENGUE.csv')
mask = df['SEM_NOT'] < 200000
df.loc[mask, 'SEM_NOT'] += 200000
df = df[df['SEM_NOT'] > 200952]
df = df[df['SEM_NOT'] < 202301]
df.to_csv('../dados_brutos/DENGUE.csv',index=False)
# Ainda com o acima aberto, faz esse laco pra separar por estado
for uf in UFCodes.keys():
    df0 = df[df['SG_UF_NOT'] == uf]
    df0.to_csv('../dados_brutos/b/'+UFCodes[uf]+'.csv',index=False)

# O do ES tem que fazer isso manualmente, nao rola nesse laco (?)
# refiz e funciona sim, ignora o comentário acima
# Abaixo, é feita a agregação por semana
for uf in UFCodes.keys():
    df0 = pd.read_csv('../dados_brutos/b/'+UFCodes[uf]+'.csv')
    df0 = df0.sort_values(by='SEM_NOT')
    df0 = df0.groupby(df0['SEM_NOT']).count()
    df0 = df0.drop(['SEM_PRI','SG_UF_NOT'],axis=1)
    df0 = df0.rename(columns={'DT_NOTIFIC':'casos','DT_OBITO':'obitos'})
    # for s in semanas:
    #     if s not in df0['semana'].values:
    #         df0.append({'semana':s,'casos':0,'obitos':0},ignore_index=True)
    # df0 = df0.sort_values('semana')
    df0.to_csv('../dados_brutos/c/'+UFCodes[uf]+'.csv',index=True)



df = pd.read_csv('../dados_brutos/c/RO.csv')
df = df.rename(columns={'SEM_NOT':'semana'})

# Finalmente é só trocar o nome da coluna das semanas, inserir semanas que faltam
for uf in UFCodes.keys():
    df = pd.read_csv('../dados_brutos/c/'+UFCodes[uf]+'.csv')
    df = df.rename(columns={'SEM_NOT':'semana'})
    for s in semanas:
        if s not in df['semana'].values:
            new_line = {'semana':s,'casos':0,'obitos':0}
            df.loc[len(df)] = new_line
    df = df.sort_values('semana')    
    df.to_csv('../dados_brutos/d/'+UFCodes[uf]+'.csv',index=False)
    
# Contagem por semana pra casos e obitos
folder_path = 'C:\\Users\\joaom\\Desktop\\tcc\\dados\\dados_em_tratamento\\xablau2\\'
csv_files = [file for file in os.listdir(folder_path)]
dfs = []
for file in csv_files:
    file_path = os.path.join(folder_path,file)
    df = pd.read_csv(file_path)
    mask = df['SEM_NOT'] < 200000
    df.loc[mask, 'SEM_NOT'] += 200000
    df_weekly = df.groupby(df['SEM_NOT']).count()
    dfs.append(df_weekly)
    df_weekly.to_csv('../dados_em_tratamento/xablau/'+file,index=True)

#==============================================================================
# ESQUEMA PRA GRAFICO DE TUDO OS CASO

df_nilso['casos'].sum()

folder_path = 'C:\\Users\\joaom\\Desktop\\tcc\\dados\\dados_versao_final\\doencas\\'
csv_files = [file for file in os.listdir(folder_path)]
dfs = []
for file in csv_files:
    file_path = os.path.join(folder_path,file)
    df = pd.read_csv(file_path)
    dfs.append(df)
df_nilso = pd.concat(dfs)
df_nilso = df_nilso.groupby('semana').agg({'casos':'sum','obitos':'sum'})
df_nilso = df_nilso.drop([200952])

folder_path2 = 'C:\\Users\\joaom\\Desktop\\tcc\\dados\\dados_versao_final\\doencas_so_confirmados\\'
csv_files2 = [file for file in os.listdir(folder_path)]
csv_files2.remove('RR.csv')
dfs2 = []
for file in csv_files2:
    file_path = os.path.join(folder_path2,file)
    df = pd.read_csv(file_path)
    dfs2.append(df)
df_nilso2 = pd.concat(dfs2)
df_nilso2 = df_nilso2.groupby('semana').agg({'casos':'sum','obitos':'sum'})
df_nilso2 = df_nilso2.drop([200952])

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(df_nilso.index,df_nilso['casos'],color='blue',label='Total de casos relatados')
ax.plot(df_nilso.index,df_nilso2['casos'],color='red',label='Casos marcados como confirmados')
ax.set_xlabel('Semana epidemiológica')
ax.set_ylabel('Número de casos')
ax.set_title('Casos de dengue no Brasil, entre 2010 e 2022')
ax.legend()
plt.grid(True)
plt.show()

#==============================================================================

peido = df0.groupby(df0['DT_NOTIFIC']).count()


moita = pd.read_csv('../dados_em_tratamento/xablau/ES.csv')
peido = moita.groupby(moita['SEM_NOT']).count()
l = []
for s in semanas:
    if s not in moita['SEM_NOT'].values:
        l.append(s)


l = []
for s in semanas:
    if s not in moita['SEM_NOT'].values:
        l.append(s)
        
print(201001 in moita['SEM_NOT'])

#==============================================================================
# SEPARANDO POR UF SÓ
#==============================================================================
# Lendo e tirando colunas extra
# Os de doenca tao com duas colunas inuteis pq eu sou burro
# df = pd.read_csv('../dados_tratamento/csv_doencas_resumidos/chik15.csv', index_col='DT_NOTIFIC', parse_dates=True)
df = pd.read_csv('../dados_tratamento/csv_doencas_resumidos/dengue22.csv')
# df = df.drop('Unnamed: 0.1', axis=1)
df = df.drop('Unnamed: 0', axis=1)
df = df.drop('ID_AGRAVO', axis=1)
df = df.drop('SG_UF', axis=1)
df = df.drop('TP_NOT', axis=1)
# Remover valores invalidos de semana
df = df[df['SEM_NOT'] > 201000]

# É aqui que separa e salva bonitinho
for uf in np.sort(df['SG_UF_NOT'].unique()):
    df0 = df[df['SG_UF_NOT'] == uf]
    df0.to_csv(path_or_buf='./doencas/dengue_'+UFCodes[uf]+'_2022.csv')

df_weekly = df.groupby(df['SEM_NOT']).count()
print(df['SG_UF_NOT'].unique())

for uf in np.sort(df['SG_UF_NOT'].unique()):
    df0 = df[df['SG_UF_NOT'] == uf]
    df0.to_csv(path_or_buf='./doencas/dengue_'+UFCodes[uf]+'_2020.csv')

df_weekly = df.groupby(df['SEM_NOT']).count()

# PARA 2020 TAVA TUDO ERRADO O CSV ????
df = pd.read_csv('../dados_em_tratamento/xablau/DENGBR20.csv.csv')
df0 = df[['DT_NOTIFIC','SEM_PRI','SG_UF_NOT','DT_OBITO']].copy()
df0 = df0[df0['SEM_PRI'] > 202000]
df0['SEM_PRI'] = df0['SEM_PRI'].astype(int)
df0 = df0.rename(columns={'SEM_PRI':'SEM_NOT'})
df0.to_csv(path_or_buf='../dados_em_tratamento/xablau/DENGBR20.csv')

print(df0['SG_UF_NOT'].unique())


# Pega a contagem de casos/dia
daily_cases = df['TP_NOT'].groupby(df.index).count()
# Pega a contagem de casos/semana
weekly_cases = df['TP_NOT'].groupby(df['SEM_NOT']).count()
# Unir por semanas
# week_by_index = df.resample('W').sum()['TP_NOT']

# Verificacao se tem numero invalido de semana
test = df['SEM_NOT'].unique()
import numpy as np
test = np.sort(test)

#==============================================================================
# Concatenando tudo os de cada ano
#==============================================================================

for uf in UFCodes.values():
    dfs = []
    for year in years:
        df = pd.read_csv('./doencas_tudocagado/'+year+'/dengue_'+uf+'_'+year+'.csv')
        dfs.append(df)
    result_df = pd.concat(dfs,ignore_index=True)
    result_df.to_csv('./doencas/'+uf+'.csv')

# ES nao tem de 2020 para cima
missing = {53: 'DF', 52: 'GO', 50: 'MS', 52: 'MT', 41: 'PR', 33: 'RJ', 43: 'RS', 42: 'SC', 35: 'SP'}

#==============================================================================
# AQUI EU SO TAVA ORDENANDO PRA NAO FICAR TAO FEIO OS ARQUIVO
#==============================================================================

for uf in UFCodes.values():
    df = pd.read_csv('./doencas_PARTEDOIS/'+uf+'.csv')
    df = df.drop('Unnamed: 0.1', axis=1)
    df = df.drop('Unnamed: 0', axis=1)
    df = df.sort_values(by='SEM_NOT')
    df.to_csv('./doencas_PARTETRES/'+uf+'.csv')

#==============================================================================
# JUNTAR POR SEMANA FINALMENTE
#==============================================================================

for uf in UFCodes.values():
    df = pd.read_csv('./doencas_PARTEDOIS/'+uf+'.csv')
    weekly_cases = df.groupby(df['SEM_NOT']).count()
    weekly_cases = weekly_cases.rename(columns={'SEM_NOT':'Semana'})
    weekly_cases = weekly_cases.rename(columns={'Unnamed: 0':'Casos'})
    weekly_cases = weekly_cases.rename(columns={'DT_OBITO':'Obitos'})
    df0 = weekly_cases[['Casos','Obitos']].copy()
    df0.to_csv('./doencas/'+uf+'.csv')
#==============================================================================
# FAZER GRAIFCO
#==============================================================================
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

df_d = pd.read_csv('./dados_versao_final/doencas/MG.csv')

fig, ax = plt.subplots(figsize=(10, 6)) # Tamanho do grafico

# Para uso de data no eixo x
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

ax.plot(df_m.index,df_d['Casos'], color='red', label='Casos')
#ax.plot(df['Semana'],df['temp_min'], color='blue', label='Minimas')
#ax.set_xlabel('Semana')
ax.set_ylabel('Casos relatados')
ax.set_title('Relatos de dengue por semana')
ax.grid(True)
ax.legend()
plt.tight_layout()

# plt.xticks(rotation=45) # Inclina legenda pra facilitar a leitura
plt.show()

