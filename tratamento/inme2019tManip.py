import pandas as pd
import numpy as np
import glob
import os
import math

def inmetD(directory,uf,year):

    # De 2019 em diante mudou um pouco o formato das coisas, entao achei menos mao criar um arquivo novo pra isso
    # Aproveitar o anterior seria mais otimizado mas o código ficaria muito feio

    csv_files = glob.glob(os.path.join(directory, "*.csv"))

    for file in csv_files:
        '''
        Para cada dia: 
        > a data né (coluna 'Data')
        > temperatura média (coluna 'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)')
        > temperatura máxima (coluna 'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)')
        > temperatura mínima (coluna 'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)') -> ignorar os valores -9999
        > precipitação média (coluna 'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)')
        > umidade relativa média (coluna 'UMIDADE RELATIVA DO AR, HORARIA (%)') -> nem sei se faz sentido isso, talvez tirar depois
        > velocidade média vento (coluna 'VENTO, VELOCIDADE HORARIA (m/s)')
        '''
        new = {
            'data' : [],
            'temp_media' : [],
            'temp_max' : [],
            'temp_min' : [],
            'prec_media' : [],
            'umid_media' : [],
            'vento_medio' : []
        }
        
        # A leitura dos csv do inmet nao funcionou com o encoding padrao, entao foi com esse latin-1
        # Mesmo após essa mudança, eu precisava antes retirar os cabeçalhos
        df = pd.read_csv(file,encoding='latin-1',sep=';')

        # Quando algum campo não foi feita a leitura, ele ta vazio; aqui pego a quantidade de campos vazios por linha
        invalid_lines = df.apply(lambda row: row.isna().sum(), axis=1)
        # Eu defini como linha lixo aquelas que têm pelo menos cinco campos vazios
        for i in range(len(invalid_lines)-1,0,-1):
            if invalid_lines[i] >= 5:
                df.drop(i, inplace=True)

        for d in df['Data'].unique():
            # Bota todas as linhas de um dia específico em um df auxiliar
            dfDay = df[df['Data'] == d]

            # Inserindo os dados no dicionário por ordem:
            new['data'].append(d)

            # Também precisa verificar para cada um se há pelo menos um valor válido
            if dfDay['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'].isna().all():
                new['temp_media'].append(0)
            else:
                # dfDay['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'] = df['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'].apply(lambda x: float(x.replace(',','.')) if isinstance(x, str) else x)
                l = []
                for t in dfDay['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)']:
                    if type(t) == str:
                        l.append(float(t.replace(',','.')))
                    elif not math.isnan(t):
                        l.append(t)
                new['temp_media'].append(round(np.mean(l),2))

            if dfDay['TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)'].isna().all():
                new['temp_max'].append(0)
            else:
                # dfDay['TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)'] = df['TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)'].apply(lambda x: float(x.replace(',','.')) if isinstance(x, str) else x)
                l = []
                for t in dfDay['TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)']:
                    if type(t) == str:
                        l.append(float(t.replace(',','.')))
                    elif not math.isnan(t):
                        l.append(t)
                new['temp_max'].append(np.max(l))

            if dfDay['TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)'].isna().all():
                new['temp_min'].append(0)
            else:
                # dfDay['TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)'] = df['TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)'].apply(lambda x: float(x.replace(',','.')) if isinstance(x, str) else x)
                l = []
                for t in dfDay['TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)']:
                    if type(t) == str:
                        l.append(float(t.replace(',','.')))
                    elif not math.isnan(t):
                        l.append(t)
                new['temp_min'].append(np.min(l))
            
            if dfDay['PRECIPITAÇÃO TOTAL, HORÁRIO (mm)'].isna().all():
                new['prec_media'].append(0)
            else:
                # dfDay['PRECIPITAÇÃO TOTAL, HORÁRIO (mm)'] = df['PRECIPITAÇÃO TOTAL, HORÁRIO (mm)'].apply(lambda x: float(x.replace(',','.')) if isinstance(x, str) else x)
                l = []
                for t in dfDay['PRECIPITAÇÃO TOTAL, HORÁRIO (mm)']:
                    if type(t) == str:
                        l.append(float(t.replace(',','.')))
                    elif not math.isnan(t):
                        l.append(t)
                new['prec_media'].append(round(np.mean(l),2))

            if dfDay['UMIDADE RELATIVA DO AR, HORARIA (%)'].isna().all():
                new['umid_media'].append(0)
            else:
                # dfDay['UMIDADE RELATIVA DO AR, HORARIA (%)'] = df['UMIDADE RELATIVA DO AR, HORARIA (%)'].apply(lambda x: float(x.replace(',','.')) if isinstance(x, str) else x)
                # new['umid_media'].append(round(np.mean([t for t in dfDay['UMIDADE RELATIVA DO AR, HORARIA (%)'] if not math.isnan(t)]),2))
                l = []
                for t in dfDay['UMIDADE RELATIVA DO AR, HORARIA (%)']:
                    if type(t) == str:
                        l.append(float(t.replace(',','.')))
                    elif not math.isnan(t):
                        l.append(t)
                new['umid_media'].append(round(np.mean(l),2))

            if dfDay['VENTO, VELOCIDADE HORARIA (m/s)'].isna().all():
                new['vento_medio'].append(0)
            else:
                # dfDay['VENTO, VELOCIDADE HORARIA (m/s)'] = df['VENTO, VELOCIDADE HORARIA (m/s)'].apply(lambda x: float(x.replace(',','.')) if isinstance(x, str) else x)
                l = []
                for t in dfDay['VENTO, VELOCIDADE HORARIA (m/s)']:
                    if type(t) == str:
                        l.append(float(t.replace(',','.')))
                    elif not math.isnan(t):
                        l.append(t)
                new['vento_medio'].append(round(np.mean(l),2))
        print('.',end='')
        pd.DataFrame.to_csv(pd.DataFrame.from_dict(new), path_or_buf='./' + year + '/' + uf + '/' + file.split('\\')[-1] + '.csv')
        
# Precisa remover as primeiras oito linhas dos csv, senão o pandas nao abre eles
def removeLines(path):
    ufs = ['AC','AL','AM','AP','BA','CE','DF','ES','GO','MA','MG','MS','MT','PA','PB','PE','PI','PR','RJ','RN','RO','RR','RS','SC','SE','SP','TO']

    for u in ufs:
        folder = path + u + '/'
        for filename in os.listdir(folder):
            file = os.path.join(folder,filename)
            with open(file, "r") as f:
                lines = f.readlines()
            with open(file, "w") as f:
                for number, line in enumerate(lines):
                    if number not in range(0,8):
                        f.write(line)

def inmetManip(path,year):
    ufs = ['AC','AL','AM','AP','BA','CE','DF','ES','GO','MA','MG','MS','MT','PA','PB','PE','PI','PR','RJ','RN','RO','RR','RS','SC','SE','SP','TO']
    for u in ufs:
        inmetD(path + u + '/',u,year)
        print(u,' pronto')

# De 2019 em diante mudaram os nomes das colunas

# removeLines('../dados_inmet/2019/')

# inmetManip('../dados_inmet/2019/','2019')

# removeLines('../dados_inmet/2020/')
# inmetManip('../dados_inmet/2020/','2020')

removeLines('../dados_inmet/2021/')
inmetManip('../dados_inmet/2021/','2021')

removeLines('../dados_inmet/2022/')
inmetManip('../dados_inmet/2022/','2022')
