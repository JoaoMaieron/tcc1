import pandas as pd
import numpy as np


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
df = pd.read_csv('../dados_inmet/2010/AC/INMET_N_AC_A104_RIO BRANCO_01-01-2010_A_31-12-2010.csv',encoding='latin-1',sep=';')

# Quando algum campo não foi feita a leitura, o default é -9999
# Aqui to pegando a quantidade de default por linha pra excluir linhas lixo depois
invalid_lines = (df == '-9999').sum(axis=1)
# Eu defini como linha lixo aquelas que têm pelo menos três campos sem valor
for index, row in df.iterrows():
    if invalid_lines[index] >= 3:
        df.drop(df.index[index], inplace=True)

for d in df['DATA (YYYY-MM-DD)'].unique():
    # Bota todas as linhas de um dia específico em um df auxiliar
    dfDay = df[df['DATA (YYYY-MM-DD)'] == d]

    # Inserindo os dados no dicionário por ordem:
    new['data'].append(d)
    new['temp_media'].append(round(np.mean([float(t.replace(',','.')) for t in dfDay['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'] if t != '-9999']),2))
    new['temp_max'].append(np.max([float(t.replace(',','.')) for t in dfDay['TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)'] if t != '-9999']))
    new['temp_min'].append(np.min([float(t.replace(',','.')) for t in dfDay['TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)'] if t != '-9999']))
    new['prec_media'].append(round(np.mean([float(t.replace(',','.')) for t in dfDay['PRECIPITAÇÃO TOTAL, HORÁRIO (mm)'] if t != '-9999']),2))
    new['umid_media'].append(round(np.mean([t for t in dfDay['UMIDADE RELATIVA DO AR, HORARIA (%)']]),2))
    new['vento_medio'].append(round(np.mean([float(t.replace(',','.')) for t in dfDay['VENTO, VELOCIDADE HORARIA (m/s)'] if t != '-9999']),2))

pd.DataFrame.to_csv(pd.DataFrame.from_dict(new), path_or_buf='./A.csv')
