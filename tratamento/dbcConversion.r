# O Ministerio da Saude disponibiliza dados em formato dbc
# O pacote read.dbc converte de dbc para csv
library(read.dbc)

# Procedimento para leitura do arquivo dbc para um dataframe
dengue10 <- read.dbc("DENGBR10.dbc")
dengue11 <- read.dbc("DENGBR11.dbc")
dengue12 <- read.dbc("DENGBR12.dbc")
dengue13 <- read.dbc("DENGBR13.dbc")
dengue14 <- read.dbc("DENGBR14.dbc")

# str(chik15)
# summary(chik15)

# Exportar o dataframe para um csv
write.csv(dengue10, "DENGBR10.csv")
write.csv(dengue11, "DENGBR11.csv")
write.csv(dengue12, "DENGBR12.csv")
write.csv(dengue13, "DENGBR13.csv")
write.csv(dengue14, "DENGBR14.csv")
