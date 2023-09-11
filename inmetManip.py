import pandas as pd
import numpy as np

# A leitura dos csv do inmet nao funcionou com o encoding padrao, entao foi com esse latin-1
# Mesmo após essa mudança, eu precisava antes retirar os cabeçalhos
df = pd.read_csv('../dados_inmet/2010/AC/INMET_N_AC_A104_RIO BRANCO_01-01-2010_A_31-12-2010.csv',encoding='latin-1',sep=';')

'''
Para cada dia: 
> a data né (coluna 'DATA (YYYY-MM-DD)')
> temperatura média (coluna 'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)')
> temperatura máxima (coluna 'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)')
> temperatura mínima (coluna 'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)') -> ignorar os valores -9999
> precipitação média (coluna 'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)')
> umidade relativa média (coluna 'UMIDADE RELATIVA DO AR, HORARIA (%)') -> nem sei se faz sentido isso, talvez tirar depois
> vento máxima (coluna 'VENTO, VELOCIDADE HORARIA (m/s)')
'''
new = {
    'data' : [],
    'temp_media' : [],
    'temp_max' : [],
    'temp_min' : [],
    'prec_media' : [],
    'umid_media' : [],
    'vento_max' : []
}

a=3 
for d in df['DATA (YYYY-MM-DD)'].unique():
    # Pega um dia específico em um df auxiliar
    dfDay = df[df['DATA (YYYY-MM-DD)'] == '2010-01-01']
    # Inserindo os dados no dicionário por ordem 
    new['data'].append(d)
    new['temp_media'].append(np.mean([float(t.replace(',','.')) for t in dfDay['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'] if t != '-9999']))
    new['temp_max'].append(np.max([float(t.replace(',','.')) for t in dfDay['TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)'] if t != '-9999']))
    new['temp_min'].append(np.min([float(t.replace(',','.')) for t in dfDay['TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)'] if t != '-9999']))
    new['prec_media'].append(np.mean([float(t.replace(',','.')) for t in dfDay['PRECIPITAÇÃO TOTAL, HORÁRIO (mm)']]))
    # new['umid_media'].append()





# Cálculo da media da temp.
# Precisa substituir as vírgulas por pontos pra que o python consiga converter cada string pra float
# new['temp_media'] = [float(t.replace(',','.')) for t in df['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'] if t != '-9999']

# Acessar por dia específico: df.loc[df['DATA (YYYY-MM-DD)'] == '2010-01-01']



