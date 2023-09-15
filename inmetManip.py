import pandas as pd
import numpy as np
import glob
import os

ufs = ['AC','AL','AM','AP','BA','CE','DF','ES','GO','MA','MG','MS','MT','PA','PB','PE','PI','PR','RJ','RN','RO','RR','RS','SC','SE','SP','TO']

calendar = [
    ('2010/01/03','2011/01/01'),
    ('2011/01/01','2011/12/31'),
    ('2012/01/01','2012/12/29'),
    ('2012/12/30','2013/12/28'),
    ('2013/12/29','2015/01/03'),
    ('2015/01/04','2016/01/02'),
    ('2016/01/03','2016/12/31'),
    ('2017/01/01','2017/12/30'),
    ('2017/12/31','2018/12/29'),
    ('2018/12/30','2019/12/28'),
    ('2019/12/29','2021/01/02'),
    ('2021/01/03','2022/01/01'),
    ('2022/01/02','2022/12/31')
]


def createFolders():
    for u in ufs:
        os.mkdir('C:\\Users\\joaom\\OneDrive\\Área de Trabalho\\tcc\\dados_tratamento\\csv_inmet\\novos'+'\\'+u)

def concatCSV(directory,uf):
    '''
    directory = formato './csv_inmet/novos/
    uf = formato 'AM'
    '''
    # Para cada estado juntar tudo os csv de mesmo local - verifica pelos 16 primeiros caracteres do filename
    # - abre a pasta de 2010;
    # - pra cada arquivo nela, avançar 2011 pra diante procurando pelo mesmo filename e concatenar se encontrar
    # - importante pegar todos os anos que a estação existe, botar no filename do csv final (trata os caso qque a estação fechou)
    
    years = ['2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021','2022']
    checked = [] # Cria uma lista pra guardar os arqs que já checou
    for y in years:
        yearsahead = [t for t in years if int(t) > int(y)]
        
        #'./csv_inmet/novos/
        # Pega todos os csv da pasta
        # csv_files_ini = glob.glob(os.path.join(directory + uf + '/', "*.csv"))
        
        # Pra cada ano vou abrir a lista dos arqs que contém
        csv_files_ini = glob.glob(os.path.join(directory + y + '/' + uf + '/', "*.csv"))
        for file in csv_files_ini:
            filename = file.split('\\')[-1][:16] # Os 16 primeiros caracteres são o que identifica cada estação
            
            # Só faço alguma coisa se o arq atual não foi verificado ainda, o que significa uma estação que abriu pós 2010
            if filename not in checked:
                yrs_available =  '_' + y + '_'
                df = pd.read_csv(file).drop(['Unnamed: 0'],axis=1)
                for y2 in yearsahead:
                    # Pra cada ano adiante vai na pasta dele e pega tudo os nome de arq
                    csv_files_y = glob.glob(os.path.join('./csv_inmet/original/'+ y2 + '/' + uf + '/', '*.csv')) 
                    for f in csv_files_y:
                        if f.split('\\')[-1][:16] == filename: # Caso encontra a estação correspondente, concatena os dataframe
                            df2 = pd.read_csv(f).drop(['Unnamed: 0'],axis=1)
                            df = pd.concat([df,df2],axis=0)
                            yrs_available = yrs_available + y2 + '_'
                checked.append(filename) # Insere o arq na lista dos que já foi
                pd.DataFrame.to_csv(df,path_or_buf='./csv_inmet/novos/'+ uf + '/' + filename + yrs_available + '.csv')

        # './csv_inmet/original/2010/AC/'
        # './csv_inmet/novos/2010/AC/'

for u in ['AC','AL','AM','AP','BA','CE','DF','ES','GO','MA','MG','MS','MT','PA','PB','PE','PI','PR','RJ','RN','RO','RR','RS','SC','SE','SP','TO']:
    concatCSV('./csv_inmet/original/',u)

# createFolders()

def inmetManip(directory,uf,year):

    # até GO funcionou
    # alteracao uma linha, até MA pronto
    # alteracao uma linha, até MG pronto

    # Pega todos os csv da pasta
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
    for u in ufs:
        inmetManip(path + u + '/',u,year)
        print(u,' pronto')

# concatCSV('./csv_inmet/original/2010/','AC')

# De 2019 em diante mudaram os nomes das colunas

# removeLines('../dados_inmet/2019/')
# inmetManip('../dados_inmet/2019/','2019')

# removeLines('../dados_inmet/2020/')
# inmetManip('../dados_inmet/2020/','2020')

# removeLines('../dados_inmet/2021/')
# inmetManip('../dados_inmet/2021/','2021')

# removeLines('../dados_inmet/2022/')
# inmetManip('../dados_inmet/2022/','2022')
