import pandas as pd
import numpy as np
import glob
import os

def inmetD(directory,uf,year):

    # até GO funcionou
    # alteracao uma linha, até MA pronto
    # alteracao uma linha, até MG pronto

    csv_files = glob.glob(os.path.join(directory, "*.csv"))

    for file in csv_files:
        '''
        Para cada dia: 
        > a data né (coluna 'DATA (YYYY-MM-DD)')
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

        # Quando algum campo não foi feita a leitura, o default é -9999
        # Aqui to pegando a quantidade de default por linha pra excluir linhas lixo depois
        invalid_lines = (df == '-9999').sum(axis=1)
        # Eu defini como linha lixo aquelas que têm pelo menos três campos sem valor

        for i in range(len(invalid_lines)-1,0,-1):
            if invalid_lines[i] >= 3:
                df.drop(i, inplace=True)
        
        # for index, row in df.iterrows():
        #     if invalid_lines[index] >= 3:
        #         df.drop(df.index[index], inplace=True)

        for d in df['DATA (YYYY-MM-DD)'].unique():
            # Bota todas as linhas de um dia específico em um df auxiliar
            dfDay = df[df['DATA (YYYY-MM-DD)'] == d]

            # Inserindo os dados no dicionário por ordem:
            new['data'].append(d)

            # Também precisa verificar para cada um se há pelo menos um valor válido
            if dfDay['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'].value_counts().get('-9999',0) == dfDay['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'].count() or dfDay['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'].value_counts().get(-9999,0) == dfDay['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'].count():
                new['temp_media'].append(0)
            else:
                if type(df['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'][0]) == str: # Os do PA tava como int
                    new['temp_media'].append(round(np.mean([float(t.replace(',','.')) for t in dfDay['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'] if t != '-9999']),2))
                else:
                    new['temp_media'].append(round(np.mean([t for t in dfDay['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'] if t != -9999]),2))

            if dfDay['TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)'].value_counts().get('-9999',0) == dfDay['TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)'].count() or dfDay['TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)'].value_counts().get(-9999,0) == dfDay['TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)'].count():
                new['temp_max'].append(0)
            else:
                if type(df['TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)'][0]) == str: # Os do PA tava como int
                    new['temp_max'].append(round(np.max([float(t.replace(',','.')) for t in dfDay['TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)'] if t != '-9999']),2))
                else:
                    new['temp_max'].append(round(np.max([t for t in dfDay['TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)'] if t != -9999]),2))

            if dfDay['TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)'].value_counts().get('-9999',0) == dfDay['TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)'].count() or dfDay['TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)'].value_counts().get(-9999,0) == dfDay['TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)'].count():
                new['temp_min'].append(0)
            else:
                if type(df['TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)'][0]) == str: # Os do PA tava como int
                    new['temp_min'].append(np.min([float(t.replace(',','.')) for t in dfDay['TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)'] if t != '-9999']))
                else:
                    new['temp_min'].append(round(np.min([t for t in dfDay['TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)'] if t != -9999]),2))                
            
            if dfDay['PRECIPITAÇÃO TOTAL, HORÁRIO (mm)'].value_counts().get('-9999',0) == dfDay['PRECIPITAÇÃO TOTAL, HORÁRIO (mm)'].count() or dfDay['PRECIPITAÇÃO TOTAL, HORÁRIO (mm)'].value_counts().get(-9999,0) == dfDay['PRECIPITAÇÃO TOTAL, HORÁRIO (mm)'].count():
                new['prec_media'].append(0)
            else:
                if type(df['PRECIPITAÇÃO TOTAL, HORÁRIO (mm)'][0]) == str: # Tinha UM csv do MT que não tava como string aqui, por isso esse if
                    new['prec_media'].append(round(np.mean([float(t.replace(',','.')) for t in dfDay['PRECIPITAÇÃO TOTAL, HORÁRIO (mm)'] if t != '-9999']),2))
                else:
                    new['prec_media'].append(round(np.mean([t for t in dfDay['PRECIPITAÇÃO TOTAL, HORÁRIO (mm)'] if t != -9999]),2))

            if dfDay['UMIDADE RELATIVA DO AR, HORARIA (%)'].value_counts().get('-9999',0) == dfDay['UMIDADE RELATIVA DO AR, HORARIA (%)'].count() or dfDay['UMIDADE RELATIVA DO AR, HORARIA (%)'].value_counts().get(-9999,0) == dfDay['UMIDADE RELATIVA DO AR, HORARIA (%)'].count():
                new['umid_media'].append(0)
            else:
                new['umid_media'].append(round(np.mean([t for t in dfDay['UMIDADE RELATIVA DO AR, HORARIA (%)']]),2))

            if dfDay['VENTO, VELOCIDADE HORARIA (m/s)'].value_counts().get('-9999',0) == dfDay['VENTO, VELOCIDADE HORARIA (m/s)'].count() or dfDay['VENTO, VELOCIDADE HORARIA (m/s)'].value_counts().get(-9999,0) == dfDay['VENTO, VELOCIDADE HORARIA (m/s)'].count():
                new['vento_medio'].append(0)
            else:
                if type(df['VENTO, VELOCIDADE HORARIA (m/s)'][0]) == str: # Os do MT tavam como int e nao string
                    new['vento_medio'].append(round(np.mean([float(t.replace(',','.')) for t in dfDay['VENTO, VELOCIDADE HORARIA (m/s)'] if t != '-9999']),2))
                else:
                    new['vento_medio'].append(round(np.mean([t for t in dfDay['VENTO, VELOCIDADE HORARIA (m/s)'] if t != -9999]),2))

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

removeLines('../dados_inmet/2019/')
inmetManip('../dados_inmet/2019/','2019')

removeLines('../dados_inmet/2020/')
inmetManip('../dados_inmet/2020/','2020')

removeLines('../dados_inmet/2021/')
inmetManip('../dados_inmet/2021/','2021')

removeLines('../dados_inmet/2022/')
inmetManip('../dados_inmet/2022/','2022')
