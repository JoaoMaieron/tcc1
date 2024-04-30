# O Ministerio da Saude disponibiliza dados em formato dbc
# O pacote read.dbc converte de dbc para csv
library(read.dbc)

# str(chik15)
# summary(chik15)

# Procedimento para leitura do arquivo dbc para um dataframe
dengue10 <- read.dbc("DENGBR10.dbc")
dengue11 <- read.dbc("DENGBR11.dbc")
dengue12 <- read.dbc("DENGBR12.dbc")
dengue13 <- read.dbc("DENGBR13.dbc")
dengue14 <- read.dbc("DENGBR14.dbc")
dengue15 <- read.dbc("DENGBR15.dbc")
dengue16 <- read.dbc("DENGBR16.dbc")
dengue17 <- read.dbc("DENGBR17.dbc")
dengue18 <- read.dbc("DENGBR18.dbc")
dengue19 <- read.dbc("DENGBR19.dbc")
dengue20 <- read.dbc("DENGBR20.dbc")
dengue21 <- read.dbc("DENGBR21.dbc")
dengue22 <- read.dbc("DENGBR22.dbc")

# Exportar o dataframe para um csv
write.csv(dengue10, "DENGBR10.csv")
write.csv(dengue11, "DENGBR11.csv")
write.csv(dengue12, "DENGBR12.csv")
write.csv(dengue13, "DENGBR13.csv")
write.csv(dengue14, "DENGBR14.csv")
write.csv(dengue15, "DENGBR15.csv")
write.csv(dengue16, "DENGBR16.csv")
write.csv(dengue17, "DENGBR17.csv")
write.csv(dengue18, "DENGBR18.csv")
write.csv(dengue19, "DENGBR19.csv")
write.csv(dengue20, "DENGBR20.csv")
write.csv(dengue21, "DENGBR21.csv")
write.csv(dengue22, "DENGBR22.csv")