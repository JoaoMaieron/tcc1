import pandas as pd
import numpy as np
import glob
import os

ufs = ['AC','AL','AM','AP','BA','CE','DF','ES','GO','MA','MG','MS','MT','PA','PB','PE','PI','PR','RJ','RN','RO','RR','RS','SC','SE','SP','TO']
# ufs = ['RR']

years = [str(n) for n in range(2010,2023)]
calendar = [
    ('2010-01-03','2011-01-01'),
    ('2011-01-01','2011-12-31'),
    ('2012-01-01','2012-12-29'),
    ('2012-12-30','2013-12-28'),
    ('2013-12-29','2015-01-03'),
    ('2015-01-04','2016-01-02'),
    ('2016-01-03','2016-12-31'),
    ('2017-01-01','2017-12-30'),
    ('2017-12-31','2018-12-29'),
    ('2018-12-30','2019-12-28'),
    ('2019-12-29','2021-01-02'),
    ('2021-01-03','2022-01-01'),
    ('2022-01-02','2022-12-31')
]

def finalVersion(directory,uf):
    '''
    O objetivo é deixar um csv só por estado
    Args:
        directory (str): caminho da pasta, formato './csv_inmet/novos/
        uf (str): sigla de estado, formato 'AM'
    '''
    files = glob.glob(os.path.join(directory + uf,'*.csv'))
    df = pd.read_csv(files[0]).drop(['Unnamed: 0'],axis=1)
    # Aqui eu transformo os zero tudo em nan pra nao ferrar o cálculo da média depois
    df[df[['temp_media','temp_max','temp_min','prec_media','umid_media','vento_medio']] == 0] = np.nan
    if len(files) > 1:
        for i in range(1,len(files)):
            df0 = pd.read_csv(files[i]).drop(['Unnamed: 0'],axis=1) # Tirando a coluna inútil
            df = pd.concat([df,df0],axis=0)
        
        # Isso aqui é um jeitinho lindo de agrupar o df por data, aplicando uma função diferente por coluna
        df = df.groupby('data').agg({
            'temp_media': np.nanmean,
            'temp_max':np.nanmax,
            'temp_min':np.nanmin,
            'prec_media':np.nanmean,
            'umid_media':np.nanmean,
            'vento_medio':np.nanmean})
    # Dando aquela truncada braba pra ficar mais bonito os csv
    df[['temp_media','prec_media','umid_media','vento_medio']] = df[['temp_media','prec_media','umid_media','vento_medio']].apply(lambda x: round(number=x,ndigits=2),axis=1)
    pd.DataFrame.to_csv(df,path_or_buf='./csv_inmet/' + uf + '.csv',index=False)

def finalAAAA():
    '''
    Só chama a montagem final pra todas uf. Vem depois de concatenar os csv por estado
    '''
    for u in ufs:
        finalVersion('./csv_inmet/novos/',u)

def createFolders():
    '''
    Só pra criar denovo as pasta tudo pra cada estado.
    '''
    for u in ufs:
        os.mkdir('C:\\Users\\joaom\\Desktop\\tcc\\dados\\dados_em_tratamento\\xablau\\'+u)
        # os.mkdir('C:\\Users\\joaom\\Desktop\\tcc\\dados\\dados_versao_final\\xablau2\\'+u)

# createFolders()

def concatCSV(directory,uf):
    '''
    Args:
        directory (str): caminho da pasta, formato './csv_inmet/novos/
        uf (str): sigla de estado, formato 'AM'
    '''
    checked = [] # Lista pra guardar os arqs que já checou

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
                df = pd.read_csv(file,encoding='latin-1',sep=';')
                # df['DATA (YYYY-MM-DD)'] = df['DATA (YYYY-MM-DD)'].astype(str)
                if 'Unnamed: 0' in df.columns:
                    df = df.drop(['Unnamed: 0'],axis=1)
                for y2 in yearsahead:
                    # Pra cada ano adiante vai na pasta dele e pega tudo os nome de arq
                    csv_files_y = glob.glob(os.path.join(directory+ y2 + '/' + uf + '/', '*.csv')) 
                    for f in csv_files_y:
                        if f.split('\\')[-1][:16] == filename: # Caso encontra a estação correspondente, concatena os dataframe                            
                            df2 = pd.read_csv(f,encoding='latin-1',sep=';')
                            # df2['DATA (YYYY-MM-DD)'] = df2['DATA (YYYY-MM-DD)'].astype(str)
                            if 'Unnamed: 0' in df2.columns:
                                df2 = df2.drop(['Unnamed: 0'],axis=1)
                            df = pd.concat([df,df2],axis=0)
                            yrs_available = yrs_available + y2 + '_'
                checked.append(filename) # Insere o arq na lista dos que já foi
                pd.DataFrame.to_csv(df,path_or_buf='./xablau/'+ uf + '/' + filename + yrs_available + '.csv',index=False)

for uf in ufs:
    concatCSV('../dados_brutos/dados_inmet/', uf)

# concatCSV('../dados_brutos/dados_inmet/','AP')

def removeLines(path):
    '''
    Remove as primeiras oito linhas do csv, senao o pandas nao abre eles direito
    '''
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

def inmetAAA(path,year):
    '''
    Só chama o pre-processamento pra cada uf
    Args:
        path (str): formato '../dados_inmet/2019/'
        year (str): ano formato '2019'    
    '''
    for u in ufs:
        inmetManip(path + u + '/',u,year)
        print(u,' pronto')


#finalAAAA()

#createFolders()

# for u in ufs:
#     concatCSV('../dados_brutos/dados_inmet/',u)

# removeLines('../dados_inmet/2019/')
# inmetManip('../dados_inmet/2019/','2019')

# inmetAAA('../dados_inmet/2010/','2010')
# inmetAAA('../dados_inmet/2011/','2011')
# inmetAAA('../dados_inmet/2012/','2012')
# inmetAAA('../dados_inmet/2013/','2013')
# inmetAAA('../dados_inmet/2014/','2014')
# inmetAAA('../dados_inmet/2015/','2015')
# inmetAAA('../dados_inmet/2016/','2016')
# inmetAAA('../dados_inmet/2017/','2017')
# inmetAAA('../dados_inmet/2018/','2018')
