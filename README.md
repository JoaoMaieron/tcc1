Alguns códigos e dados utilizados em meu TCC, que envolvia o uso de LSTM, SVM e RF para previsões de casos de dengue com base em dados meteorológicos e do Google Trends.
Maiores esclarecimentos estão no texto em si, incluso neste repositório.

Aquilo que poderá ser de algum interesse a terceiros são os conteúdos da pasta dados_versao_final, na qual estão arquivos .csv contendo dados históricos de 2010 a 2022, agrupados por estado e semana epidemiológica, no seguinte formato:

Histórico de dengue - dados brutos retirados de https://datasus.saude.gov.br/transferencia-de-arquivos/
colunas: semana: código da semana epidemiológica; casos: nº de casos relatados; obitos: mortes por dengue relatadas;

Histórico meteorológico - dados brutos retirados de https://portal.inmet.gov.br/dadoshistoricos - originalmente disponíveis por estação meteorológica e com frequência horária, os dados foram aqui agrupados por estado e semana, assim como os demais dados utilizados.
colunas: data: primeiro dia da semana epidemiológica de referência; temp_max, temp_min, temp_media: máximas, mínimas e médias registradas em ºC; prec: acúmulo de precipitação em mm; umid_max, umid_min, umid_media: máximas, mínimas e médias de umidade relativa do ar, em %; vento_max, vento_media: rajadas de vento máximas e médias, em m/s;

Números do Google Trends
colunas: Week: primeiro dia da semana epidemiológica de referência; buscas: parcela de buscas envolvendo a palavra "dengue" para aquela semana;
