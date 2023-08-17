# Uso da biblioteca pytrends para baixar dados do Google trends
from pytrends.request import TrendReq

pytrend = TrendReq() # Configura a conexao com a Google

''' 
Problema dessa biblioteca é que os resultados diferem do exibido pela interface do Google.
O motivo para isso parece ser que o Google exibe diferentes resultados dependendo da sua geolocalização 
e a biblioteca te coloca no US. (Mas é muito mais rápido que baixar manualmente pelo google).
Aliás pesquisando em dias diferentes, mesmo na interface oficial, vêm resultados diferentes (não que importe muito).
'''

# Termos de busca
kw_list = ['zika', 'dengue','chikungunya'] 

# Lista dos códigos geo 
geos = ['BR','BR-AC','BR-AL','BR-AP','BR-AM','BR-BA','BR-CE','BR-DF','BR-ES','BR-GO','BR-MA','BR-MT','BR-MS','BR-MG','BR-PA','BR-PB','BR-PR','BR-PE','BR-PI','BR-RJ','BR-RN','BR-RS','BR-RO','BR-RR','BR-SC','BR-SP','BR-SE','BR-TO']

# Nome de cada estado pra organizar os csv
names = ['Brasil','Acre','Alagoas','Amapá','Amazonas','Bahia','Ceará','Distrito Federal','Espírito Santo','Goiás','Maranhão','Mato Grosso','Mato Grosso do Sul','Minas Gerais','Pará','Paraíba','Paraná','Pernambuco','Piauí','Rio de Janeiro','Rio Grande do Norte','Rio Grande do Sul','Rondônia','Roraima','Santa Catarina','São Paulo','Sergipe','Tocantins']

# Início e fim de cada ano epidemiológico que vou pesquisar, de 2010 a 2022
years = ['2010-01-03 2011-01-01',
         '2011-01-02 2011-12-31',
         '2012-01-01 2012-12-29',
         '2012-12-30 2013-12-28',
         '2013-12-29 2015-01-03',
         '2015-01-04 2016-01-02',
         '2016-01-03 2016-12-31',
         '2017-01-01 2017-12-30',
         '2017-12-31 2018-12-29',
         '2018-12-30 2019-12-28',
         '2019-12-29 2021-01-02',
         '2021-01-03 2022-01-01',
         '2022-01-02 2022-12-31']

# Os anos em si só pra organizar os arquivos
years2 = ['2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021','2022']

for g, n in zip(geos, names):
    for y, y2 in zip(years, years2):
        pytrend.build_payload(kw_list=kw_list, geo=g, timeframe=y)
        df = pytrend.interest_over_time().drop(columns='isPartial')
        df.to_csv('./aa/' + y2 + '/' + n + y2 + '.csv')

# Explicacao dos atributos em https://pypi.org/project/pytrends/#common-api-parameters
# pytrend.build_payload(kw_list=kw_list, geo='BR-AC', timeframe='2018-12-30 2019-12-28')

# Teste que fiz antes, deixa aqui pra consulta sl
# df = pytrend.interest_over_time().drop(columns='isPartial') # Essa coluna isPartial vem junto, nao precisa
# df.to_csv('./teste.csv')
# a=3
