# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 19:39:26 2024

@author: torug
"""

# Instalando pacotes necessários para análise de dados e visualização
!pip install pandas
!pip install numpy
!pip install factor_analyzer
!pip install sympy
!pip install scipy
!pip install matplotlib
!pip install seaborn
!pip install plotly
!pip install pingouin
!pip install pyshp

#%% Importando os pacotes necessários

import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio
pio.renderers.default = 'browser'
import plotly.graph_objects as go

#%% Carregando e explorando o banco de dados

# Carregando o banco de dados de preços de casas a partir de um arquivo Excel
casas = pd.read_excel('preco_casas.xlsx')

# Visualizando as primeiras linhas do dataframe
casas.head()

# Obtendo informações sobre o dataframe, como tipos de dados e valores nulos
casas.info()

# Obtendo estatísticas descritivas do dataframe
casas_describe = casas.describe()

# Calculando a matriz de correlação entre as variáveis
corr = casas.corr()

#%% Criando um gráfico de calor interativo para visualizar as correlações

fig = go.Figure()

# Adicionando o gráfico de calor
fig.add_trace(
    go.Heatmap(
        x = corr.columns,
        y = corr.index,
        z = np.array(corr),
        text = corr.values,
        texttemplate = '%{text:.3f}',
        colorscale = 'viridis'
    )
)

# Atualizando o layout do gráfico
fig.update_layout(
    height = 750,
    width = 750,
    yaxis = dict(autorange="reversed")
)

# Exibindo o gráfico
fig.show()

#%% Preparando os dados para Análise Fatorial

# Removendo a coluna 'property_value' para comparar o futuro ranking com essa variável
casas_pca = casas.drop(columns=['property_value'])

# Calculando o teste de esfericidade de Bartlett
bartlett, p_value = calculate_bartlett_sphericity(casas_pca)

print(f'Qui² Bartlett: {round(bartlett, 2)}')
print(f'p-valor: {round(p_value, 4)}')

#%% Ajustando o modelo de Análise Fatorial

# Ajustando o modelo de análise fatorial com 8 fatores
fa = FactorAnalyzer(n_factors=8, method='principal').fit(casas_pca)

# Obtendo os autovalores
autovalores = fa.get_eigenvalues()[0]
print(autovalores)

# Verificando a soma dos autovalores
print(round(autovalores.sum(), 2))

# Reajustando o modelo de análise fatorial com 3 fatores (fatores com autovalores > 1)
fa = FactorAnalyzer(n_factors=3, method='principal').fit(casas_pca)

# Obtendo a variância explicada pelos fatores
autovalores_fatores = fa.get_factor_variance()

# Formatando a tabela de variância dos autovalores
tabela_eigen = pd.DataFrame(autovalores_fatores)
tabela_eigen.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_eigen.columns)]
tabela_eigen.index = ['Autovalor','Variância', 'Variância Acumulada']
tabela_eigen = tabela_eigen.T

print(tabela_eigen)

#%% Obtendo e visualizando as cargas fatoriais

# Obtendo as cargas fatoriais
cargas_fatoriais = fa.loadings_

# Criando uma tabela com as cargas fatoriais
tabela_cargas = pd.DataFrame(cargas_fatoriais)
tabela_cargas.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_cargas.columns)]
tabela_cargas.index = casas_pca.columns

#%% Calculando e visualizando as comunalidades

# Obtendo as comunalidades
comunalidades = fa.get_communalities()

# Criando uma tabela com as comunalidades
tabela_comunalidades = pd.DataFrame(comunalidades)
tabela_comunalidades.columns = ['Comunalidades']
tabela_comunalidades.index = casas_pca.columns

print(tabela_comunalidades)

#%% Transformando os dados nos novos fatores

# Transformando os dados originais nos novos fatores
fatores = pd.DataFrame(fa.transform(casas_pca))
fatores.columns = [f"Fator {i+1}" for i, v in enumerate(fatores.columns)]

# Concatenando os novos fatores ao dataframe original
casas = pd.concat([casas.reset_index(drop=True), fatores], axis=1)

#%% Obtendo os scores dos fatores

# Obtendo os scores dos fatores
scores = fa.weights_

# Criando uma tabela com os scores dos fatores
tabela_scores = pd.DataFrame(scores)
tabela_scores.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_scores.columns)]
tabela_scores.index = casas_pca.columns

print(tabela_scores)

#%% Criação de ranking

# Inicializando a coluna de ranking com 0
casas['Ranking'] = 0

# Calculando o ranking ponderado pela variância explicada de cada fator
for index, item in enumerate(list(tabela_eigen.index)):
    variancia = tabela_eigen.loc[item]['Variância']
    casas['Ranking'] = casas['Ranking'] + casas[tabela_eigen.index[index]] * variancia

#%% Análise de correlação entre o ranking e o valor das propriedades

# Calculando a correlação de Pearson entre o ranking e a variável 'property_value'
correlacao = pg.rcorr(casas[['Ranking', 'property_value']], method='pearson', upper='pval', decimals=4, pval_stars={0.01: '***', 0.05: '**', 0.10: '*'})

print(correlacao)
