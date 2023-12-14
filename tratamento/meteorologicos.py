# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os

# UFCodes = {
#     11 : 'RO', 12 : 'AC', 13 : 'AM', 14 : 'RR', 15 : 'PA', 16 : 'AP', 17 : 'TO', 
#     21 : 'MA', 22 : 'PI', 23 : 'CE', 24 : 'RN', 25 : 'PB', 26 : 'PE', 27 : 'AL', 28 : 'SE', 29 : 'BA',
#     31 : 'MG', 32 : 'ES', 33 : 'RJ', 35 : 'SP', 
#     41 : 'PR', 42 : 'SC', 43 : 'RS',
#     50 : 'MS', 51 : 'MT', 52 : 'GO', 53 : 'DF'
# }


UFCodes = {
    14 : 'RR', 16 : 'AP'
}

years = [str(n) for n in range(2010,2023)]

df = pd.read_csv('./doencas/RS.csv')

# =============================================================================
# QUANDO EU ACHEI BOA IDEIA COLOCAR O CODIGO SEMANAL EM UMA COLUNA NOS DO INMET
# =============================================================================

# for filename in os.listdir(folder_path):
#     df = pd.read_csv(os.path.join(folder_path,filename))
#     #df = df.drop('Unnamed: 0', axis=1)
#     df = df.rename(columns={'data':'dia_inicial'})
#     if len(df) == 679:
#         df = df.drop(len(df)-1)
#     df.insert(0,'Semana',semanas)
#     df.to_csv('./a/'+filename, index=False)
# # Povoando dicionario semana : dia
# dfIndex = pd.read_csv('../dados/dados_versao_final/doencas/MG.csv')
# semanas = dfIndex['Semana']
# df = pd.read_csv('../dados/dados_versao_final/meteorologicos/RS.csv')
# dia_inicial = df['data']
# week_dict = { w : d for w, d in zip(semanas,dia_inicial)}

# seios = pd.read_csv('./doencas/2010/dengue_AC_2010.csv')
# semanas = seios['SEM_NOT'].unique()

# folder_path = '../dados_tratamento/csv_inmet/novos/'

# =============================================================================
# =============================================================================

def csvConcat(folder_path):
    '''
    Juntar todos csv de cada estado em um só
    Args:
        folder_path (str): caminho da pasta onde tao os csv
    '''
    for uf in UFCodes.values():
        dfs = []
        path = folder_path + uf + '/'
        for filename in os.listdir(path):
            if filename.endswith('.csv'):
                file_path = os.path.join(path,filename)
                df = pd.read_csv(file_path)
                dfs.append(df)
        result_df = pd.concat(dfs,ignore_index=True)
        # Retirar dias 1 e 2 Jan. 2010 pq nao tao no calendario q vou usar
        # result_df = result_df[result_df['data'] != '2010-01-01' and result_df['data'] != '2010-01-02']
        result_df.to_csv(path_or_buf='./meteorologicos/'+uf+'.csv')


# csvConcat('./xablau/')


# =============================================================================
# =============================================================================

def weeklyCsv():
    '''
    Pega os csv que tem uma linha por dia e agrega por semana    
    '''
    for uf in UFCodes.values():
        df = pd.read_csv('./meteorologicos_concatenados/'+uf+'.csv')
        # df = df.drop('Unnamed: 0.1', axis=1)
        # df = df.drop('Unnamed: 0', axis=1)
        # Retirando os dias que nao precisa
        df = df[df['data'] != '2010-01-01'] 
        df = df[df['data'] != '2010-01-02']
        df = df.sort_values(by='data')
    
        # Os cara mudaram o modelo de data de 2019 em diante pqp
        df_ate2018 = df[df['data'] < '2019-01-01']
        df_ate2018['data'] = pd.to_datetime(df_ate2018['data'], format='%Y-%m-%d', errors='coerce').dt.date
        df_pos2019 = df[df['data'] > '2018-12-31']
        df_pos2019['data'] = pd.to_datetime(df_pos2019['data'], format='%Y/%m/%d', errors='coerce').dt.date
        df_daily = pd.concat([df_ate2018,df_pos2019])
    
        # Ajustando os valores diarios
        df_daily = df_daily.groupby('data').agg({
            'temp_max': 'max',
            'temp_min': 'min',
            'temp_media': 'mean',
            'prec_media': 'mean',
            'umid_media': 'mean',
            'vento_medio': 'mean'
            }).reset_index()
    
        df_daily['data'] = pd.to_datetime(df_daily['data'], format='%Y-%m-%d', errors='coerce')
        df_daily.set_index('data',inplace=True)
        # Ajustando por semana agora
        df_weekly = df_daily.resample('W-Sun').agg({
            'temp_max': 'max',
            'temp_min': 'min',
            'temp_media': 'mean',
            'prec_media': 'mean',
            'umid_media': 'mean',
            'vento_medio': 'mean'
            }).reset_index()
    
        df_weekly['data'] = df_weekly['data'].dt.date
        df_weekly.to_csv('./meteorologicos/'+uf+'.csv')
    
#==============================================================================
# NAO ME LEMBRO MAIS O QUE ERA ISSO
#==============================================================================
# d = ['AP','DF','GO','MS','MT','PR','RS','SC','SP']
# m = ['RO','RR']
# def euQueroMorrer():
#     for uf in UFCodes.values():
#         if uf in m:
#             df = pd.read_csv('../dados/dados_versao_final/meteorologicos/'+uf+'.csv')
#             for w in week_dict.values():
#                 if w not in df['data'].values:
#                     new_row = pd.Series([w,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],index=df.columns)
#                     df.loc[len(df)] = new_row
                    
#             df = df.sort_values(by='data')
#             df.to_csv('./a/'+uf+'.csv',index=False)

#==============================================================================
# PRE PROCESSAMENTO
#==============================================================================

# Primeiro a filtragem de colunas
e = ['HORA (UTC)',
              'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)',
              'PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)',
              'PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)',
              'RADIACAO GLOBAL (KJ/m²)',
              'TEMPERATURA DO PONTO DE ORVALHO (°C)',
              'TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)',
              'TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)',
              'VENTO, DIREÇÃO HORARIA (gr) (° (gr))',
              'VENTO, VELOCIDADE HORARIA (m/s)',
              'Unnamed: 19'] #'RADIACAO GLOBAL (Kj/m²)' - essa deu problema sl pq

'''
Colunas que ficam:
    'DATA (YYYY-MM-DD)', 
    'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)',
    'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)',
    'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)',
    'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)',
    'UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)',
    'UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)',
    'UMIDADE RELATIVA DO AR, HORARIA (%)',
    'VENTO, RAJADA MAXIMA (m/s)',
    'Data', 
    'Hora UTC',
'''

# def problematicos():
#     folder_path = 'C:\\Users\\joaom\\Desktop\\tcc\\dados\\dados_em_tratamento\\dados\\'
    
#     # Primeiro mexer no formato das datas
#     nelso = ['2019','2020','2021','2022']
#     for uf in UFCodes.values():
#         csv_files = [file for file in os.listdir(folder_path+uf+'\\')]
#         for file in csv_files:
#             sexo = False
#             for n in nelso:
#                 if n in file:
#                     sexo = True
#             if sexo:
#                 file_path = os.path.join(folder_path+uf+'\\',file)
#                 df = pd.read_csv(file_path)
#                 # substituir o / por - de 2019 em diante
#                 df['data'] = df['data'].str.replace('/', '-')
#                 df.to_csv('../dados_em_tratamento/dados/'+uf+'/'+file,index=False)
#     # Segundo concatenar
#     for uf in UFCodes.values():
#         csv_files = [file for file in os.listdir(folder_path+uf+'\\')]
#         dfs = []
#         for file in csv_files:
#             file_path = os.path.join(folder_path+uf+'\\', file)
#             df = pd.read_csv(file_path).drop('Unnamed: 0', axis=1)
#             dfs.append(df)
#         result_df = pd.concat(dfs, ignore_index=True)    
#         result_df.to_csv(uf+'.csv',index=False)
#         print(uf,' PRONTO')
    
#     # Terceiro 
#     for uf in UFCodes.values():
#         df = pd.read_csv(uf+'.csv')
#         df = df[df['data'] != '2010-01-01'] 
#         df = df[df['data'] != '2010-01-02']
#         df = df.rename(columns = {                
#                 'vento_medio': 'vento',
                
#                 })

def preProcessMega():
    # Tirar as colunas que nao vao ser usadas
    folder_path = 'C:\\Users\\joaom\\Desktop\\tcc\\dados\\dados_versao_final\\xablau\\'
    for uf in UFCodes.values():
        csv_files = [file for file in os.listdir(folder_path+uf+'\\')]
        for file in csv_files:
            file_path = os.path.join(folder_path+uf+'\\',file)
            df = pd.read_csv(file_path)
            test = []
            for c in e:
                if c in df.columns:
                    test.append(c)
                    df = df.drop(c,axis=1)
            df.to_csv('../dados_versao_final/xablau/'+uf+'/'+file+'.csv',index=False)
    
    # STANDARDIZAR AS DATAS - algumas tao usando / em vez de -
    # folder_path = 'C:\\Users\\joaom\\Desktop\\tcc\\dados\\dados_versao_final\\xablau2\\'
    nelso = ['2019','2020','2021','2022']
    for uf in UFCodes.values():
        csv_files = [file for file in os.listdir(folder_path+uf+'\\')]
        for file in csv_files:
            sexo = False
            for n in nelso:
                if n in file:
                    sexo = True
            if sexo:
                file_path = os.path.join(folder_path+uf+'\\',file)
                df = pd.read_csv(file_path)
                # substituir o / por - de 2019 em diante
                df['Data'] = df['Data'].str.replace('/', '-')
                df.to_csv('../dados_versao_final/xablau/'+uf+'/'+file,index=False)
    
    # Concatenar todos por estado
    # Antes ja concatenei por estacao meteorologica, isso ta em outro arquivo
    # folder_path = 'C:\\Users\\joaom\\Desktop\\tcc\\dados\\dados_versao_final\\xablau\\'
    for uf in UFCodes.values():
        csv_files = [file for file in os.listdir(folder_path+uf+'\\')]
        dfs = []
        for file in csv_files:
            file_path = os.path.join(folder_path+uf+'\\', file)
            df = pd.read_csv(file_path)
            if 'Unnamed: 0' in df.columns:
                df = df.drop('Unnamed: 0', axis=1)
            dfs.append(df)
        result_df = pd.concat(dfs, ignore_index=True)    
        result_df.to_csv(uf+'.csv',index=False)
        print(uf,' PRONTO')
    
    # Colocando todas as datas em uma mesma coluna
    for uf in UFCodes.values():
        df = pd.read_csv(uf+'.csv')
        datas = df['DATA (YYYY-MM-DD)'].dropna().tolist()
        datas = datas + df['Data'].dropna().tolist()
        df.insert(0,'data',datas)
        df.to_csv('../dados_versao_final/xablau/'+uf+'.csv',index=False)
    
    # Retirar 2010 Jan. 1 e 2
    # Substituir as virgulas por pontos nos floats
    # Substituir os -9999 por nan
    # Agrupar por dia e depois por semana
    for uf in UFCodes.values():
        df = pd.read_csv('../dados_versao_final/xablau/'+uf+'.csv')
        df = df[df['data'] != '2010-01-01'] 
        df = df[df['data'] != '2010-01-02']
        for c in df.columns[1:]:
            if c != 'DATA (YYYY-MM-DD)' and c != 'Data' and c != 'Hora UTC' and c != 'HORA (UTC)':
                df[c] = df[c].astype(str)
                df[c] = df[c].str.replace(',', '.').astype(float)
                df[c] = df[c].replace(-9999, np.nan)
        df_daily = df.groupby('data').agg({
                'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)': 'max',
                'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)': 'min',
                'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)': 'mean',
                'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)': 'sum',
                'UMIDADE RELATIVA DO AR, HORARIA (%)': 'mean',
                'VENTO, RAJADA MAXIMA (m/s)': 'mean'
                }).reset_index()
        df_daily = df_daily.rename(columns = {
                'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)': 'temp_max',
                'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)': 'temp_min',
                'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)': 'temp_media',
                'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)': 'prec',
                'UMIDADE RELATIVA DO AR, HORARIA (%)': 'umid_media',
                'VENTO, RAJADA MAXIMA (m/s)': 'vento'
                })
        df_daily['data'] = pd.to_datetime(df_daily['data'])
        df_daily.set_index('data', inplace=True)
        df_weekly = df_daily.resample('W-Sun').agg({
                'temp_max': 'max',
                'temp_min': 'min',
                'temp_media': 'mean',
                'prec': 'mean',
                'umid_media': 'mean',
                'vento': 'mean'
                }).reset_index()
        df_weekly.to_csv(uf+'.csv')

    # Ao final de tudo isso, RR ainda ficou com algumas semanas faltando
    # for uf in UFCodes.values():
    #     df = pd.read_csv('../dados_em_tratamento/xablau2/'+uf+'.csv')
    #     if df.shape[0] != 679:
    #         print(uf,df.shape)

# preProcessMega()

# df = pd.read_csv('RR.csv')

# df = df.drop(len(df)-1)
# df = df.drop('Unnamed: 0',axis=1)

# df.to_csv('RR.csv',index=False)

# df = df.dropna()


#==============================================================================
# FAZER GRAIFCO
#==============================================================================         
            
import matplotlib.pyplot as plt

# Mais simplezin
df = pd.read_csv('../dados/dados_versao_final/meteorologicos/AC.csv')
plt.plot(df['Semana'],df['temp_max'],color='red',label='Maximas')
plt.plot(df['Semana'],df['temp_min'],color='blue',label='Minimas')
plt.xlabel('Semanas')
plt.ylabel('Temps. em ºC')
plt.legend()
plt.show()

# ChatGPT, mais gabaritado
fig, ax = plt.subplots(figsize=(10, 6)) # Tamanho do grafico
ax.plot(df['Semana'],df['temp_max'], color='red', label='Maximas')
ax.plot(df['Semana'],df['temp_min'], color='blue', label='Minimas')
ax.set_xlabel('Semana')
ax.set_ylabel('Temp. em ºC')
ax.set_title('Temperaturas máximas e mínimas por semana')
ax.legend()

# plt.xticks(rotation=45) # Inclina legenda pra facilitar a leitura
# ax.legend('ASSA',loc='upper left')
plt.show()