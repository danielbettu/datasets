# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 11:01:02 2025

@author: Daniel Bettú

Cálculo de gradientes de pressão e tensões a partir de dados sônicos para montagem de janela operacional - SITUAÇÃO ONSHORE

PARA SIMPLIFICAÇÃO, CONSIDERA NÍVEL DA ÁGUA NO TERRENO = 0 m
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import math
import statistics

# Carregar dados do arquivo de entrada em txt
# df_vagarosidade = pd.read_csv('https://raw.githubusercontent.com/danielbettu/datasets/refs/heads/main/DADOS_1_onshore_MM.csv', sep= "\t", header=0)
# df_dados_insitu = pd.read_csv('https://raw.githubusercontent.com/danielbettu/datasets/refs/heads/main/tensoes_insitu_poisson_MM_ONSHORE.csv', sep= ";", header=0)
df_vagarosidade = pd.read_csv('/home/bettu/Downloads/DADOS_9_onshore_MM.csv', sep= "\t", header=0)
df_dados_insitu = pd.read_csv('/home/bettu/Downloads/tensoes_insitu_poisson_MM_ONSHORE.csv', sep=",", header=0)

# Criando o dataframe 'dados' com os dados numéricos
dados = pd.merge(df_vagarosidade, df_dados_insitu, on='Prof', how='inner')

 
plt.figure(figsize=(18, 10))
plt.plot(dados['Prof'], dados['DT'])
plt.xlabel('Profundidade (m)')
plt.ylabel('Vagarosidade (uS/ft)')
plt.grid(which='both', linestyle='--', linewidth=0.5, axis='both', color='gray')
plt.show()
plt.close()
 
#####################################
# CORRELACAO DE GARDNER - estimativa da densidade da Fm

# Cálculo da correlação de Gardner 
def perform_gardner_correl(dados_vagarosidade, coef_a, coef_b):
    # Realizar a regressão linear
    # x = dados_vagarosidade  # valores converte em numpy array, -1 significa que calcula a dimensão de linhas, mas tem uma coluna

    dens_formacao = coef_a * (1000000 / dados_vagarosidade) ** coef_b

    return dens_formacao


# Definição das variáveis para correlacao de Gardner
coef_a = 0.234  # coef a de Gardner
coef_b = 0.25  # coef b de Gardner

# Calcular a correlação de Gardner
dens_formacao = perform_gardner_correl(dados.iloc[:, 1], coef_a, coef_b)
dens_formacao.name = 'densidade'

profundidade = dados.loc[:, "Prof"]

 
plt.figure(figsize=(18, 10))
plt.plot(profundidade, dens_formacao)
plt.grid(which='both', linestyle='--', linewidth=0.5, axis='both', color='gray')
plt.xlabel('Profundidade (m)')
plt.ylabel('Densidade da Formação (g/cm$^{3}$)')
plt.show()

# Limpar a tela de plotagem
plt.clf()
 
###################
# GRADIENTE DE SOBRECARGA

# cálculo de espessuras
# Calcular a diferença entre a linha atual e a próxima
espessura_camada = profundidade.diff()
espessura_camada.name = 'espessura'
# A primeira linha não tem uma linha anterior, então vamos preencher o NaN com o valor 0
espessura_camada.iloc[0] = (profundidade.iloc[0])
                      
# Realizar o cálculo do gradiente de sobrecarga
tensao_camada = 1.422 * (espessura_camada * dens_formacao)
tensao_camada.name = 'tensao_camada'

tensao_sobrecarga = tensao_camada.cumsum()
tensao_sobrecarga.name = 'tensao_sobrecarga'
gradiente_sobrecarga = tensao_sobrecarga / (0.1704 * profundidade)
gradiente_sobrecarga.name = 'Gov'

# Plotar
plt.figure(figsize=(18, 10))
plt.plot(profundidade, tensao_sobrecarga)
plt.xlabel('Profundidade (m)')
plt.ylabel('Tensão de sobrecarga (psi)')
plt.grid(which='both', linestyle='--', linewidth=0.5, axis='both', color='gray')
plt.show()

plt.close() 

# Plotar
plt.figure(figsize=(18, 10))
plt.plot(profundidade, gradiente_sobrecarga)
plt.xlabel('Profundidade (m)')
plt.ylabel('Gradiente sobrecarga (lbf/gal)')
plt.grid(which='both', linestyle='--', linewidth=0.5, axis='both', color='gray')
plt.show()

plt.close()
 
#################### 
# ESTIMATIVA DE GRADIENTE DE PRESSÃO DE POROS
# Aplicação da Equação de Eaton para estimar o Gp

# Definição das variáveis
dens_agua = 1.03  # g/cm3
lamina_agua = 0  # m

# Cálculo da tendência linear de compactação
# definição do topo da zona superpressurizada - zona de desequilíbrip de subcompactação
topo_subcomp = float(input("Qual é a profundidade de início da subcompactação? "))
# topo_subcomp = 2000
print("Profundidade de início da subcompactação:", topo_subcomp, ' m')

# Encontrar o índice da menor profundidade até 'topo_subcomp'
indices = (dados.iloc[:, 0] <= topo_subcomp)

prof_regression = dados.loc[indices, dados.columns[0]]  # cria uma série com as profundidades menores que topo_subcomp
# Cria uma nova série ignorando o valor da primeira linha
prof_regression.modificada = prof_regression.iloc[1:]
prof_regression = prof_regression.modificada

DT_regression = dados.loc[indices, dados.columns[1]]  # cria uma série com DT para as profundidades menores que topo_subcomp
DT_regression_modificada = DT_regression.iloc[1:] # Cria uma nova série ignorando o valor da primeira linha
DT_regression = DT_regression_modificada

# Realizar a regressão linear
slope, intercept = statistics.linear_regression(prof_regression, DT_regression)
# Imprimir na tela a equação da reta
print("A equação da linha de tendência é: Vagarosidade = ", slope, " * prof ", " + ", intercept)


# Reshape your data
deltaT_medido = dados["DT"]

# Use the DataFrame for prediction
deltaT_esperado = (slope * profundidade) + intercept
deltaT_esperado.iloc[0] = np.nan # excluindo valor atribuído para a primeira profundidade (mâmina dágua)
diff_deltaT = deltaT_medido - deltaT_esperado

 
plt.close()
plt.figure(figsize=(18, 10))
plt.plot(profundidade, diff_deltaT)
plt.xlabel('Profundidade (m)')
plt.ylabel('Diferença da Vagarosidade medida X estimada ($\\mu$S/ft)')
plt.grid(which='both', linestyle='--', linewidth=0.5, axis='both', color='gray')
plt.show()
plt.close()

# Plotar dados e reta de tendência
plt.figure(figsize=(18, 10))
plt.plot(dados.iloc[:, 0], dados.iloc[:, 1], label='Dados originais')
plt.plot(profundidade, deltaT_esperado, color='red',
         label='Linha de tendência')  # Adicionar a linha de tendência ao gráfico
plt.xlabel('Profundidade (m)')
plt.ylabel('Vagarosidade estimada ($\\mu$S/ft)')
plt.grid(which='both', linestyle='--', linewidth=0.5, axis='both', color='gray')
plt.show()
 

plt.close()

###################
# Estimativa de gradiente de pressão de poros - gradiente_pressao_poros

pressao_teorica_agua = 1.422*(profundidade * dens_agua)
gradiente_teorico_agua = pressao_teorica_agua / (0.1704 * profundidade)
gradiente_teorico_agua.iloc[0] = 0 # atribuindo o valor para a primeira profundidade
gradiente_sobrecarga.iloc[0] = 0 # atribuindo o valor para a primeira profundidade

# gradiente_pressao_poros = pd.Series([np.nan] * len(profundidade), index=profundidade)
for i in range(len(profundidade)):
    gradiente_pressao_poros = gradiente_sobrecarga - ((gradiente_sobrecarga - gradiente_teorico_agua) * ((deltaT_esperado / deltaT_medido) ** 2))
        
pressao_poros_estimada = gradiente_pressao_poros * 0.1704 * profundidade

plt.figure(figsize=(18, 10))
plt.plot(profundidade, pressao_poros_estimada)
plt.xlabel('Profundidade (m)')
plt.ylabel('Pressão de poros estimada (psi)')
plt.grid(which='both', linestyle='--', linewidth=0.5, axis='both', color='gray')
plt.show()
plt.close()

plt.figure(figsize=(18, 10))
plt.plot(profundidade, gradiente_pressao_poros)
plt.xlabel('Profundidade (m)')
plt.ylabel('Gradiente de pressão de poros (lbf/gal)')
plt.grid(which='both', linestyle='--', linewidth=0.5, axis='both', color='gray')
plt.show()
plt.close()


###################
# Estimativa de gradiente de pressão de colapso

tensao_hor_max = dados['TH']
tensao_hor_min = dados['Th']
S0 = dados['Coesao(psi)']
phi = dados['angulo_atrito_interno']  # degrees
phi_rad = np.deg2rad(phi)
C0 = 2 * S0 * (np.cos(phi_rad) / (1 - np.sin(phi_rad)))
tan_phi2 = (np.tan((np.pi / 4) + (phi_rad / 2))) ** 2
tan_phi2.name = 'tan_phi2'
pressao_minima_colapso = ((3 * tensao_hor_max) - tensao_hor_min - C0 + (pressao_poros_estimada * (tan_phi2 - 1))) / (tan_phi2 + 1)
gradiente_colapso = pressao_minima_colapso / (0.1704 * profundidade)

 
plt.figure(figsize=(18, 10))
plt.plot(profundidade, dados['TH'])
plt.xlabel('Profundidade (m)')
plt.ylabel('TH (psi))')
plt.grid(which='both', linestyle='--', linewidth=0.5, axis='both', color='gray')
plt.show()
plt.close()

plt.figure(figsize=(18, 10))
plt.plot(profundidade, dados['Th'])
plt.xlabel('Profundidade (m)')
plt.ylabel('Th (psi))')
plt.grid(which='both', linestyle='--', linewidth=0.5, axis='both', color='gray')
plt.show()
plt.close()
 
plt.figure(figsize=(18, 10))
plt.plot(profundidade, pressao_minima_colapso)
plt.xlabel('Profundidade (m)')
plt.ylabel('Pressão mínima de colapso (psi))')
plt.grid(which='both', linestyle='--', linewidth=0.5, axis='both', color='gray')
plt.show()
plt.close()

plt.figure(figsize=(18, 10))
plt.plot(profundidade, gradiente_colapso)
plt.xlabel('Profundidade (m)')
plt.ylabel('Gradiente de Colapso (lbf/gal)')
plt.grid(which='both', linestyle='--', linewidth=0.5, axis='both', color='gray')
plt.show()
plt.close()


###################
# Estimativa de gradiente de fratura
# Gf = K (Gov – Gpp) + Gpp pag 322 Rocha e Azevedo

coef_poisson = dados['Poisson']

coef_K = coef_poisson / (1 - coef_poisson)  # pág 324 Rocha e Azevedo
gradiente_fratura = (coef_K * (gradiente_sobrecarga - gradiente_pressao_poros)) + gradiente_pressao_poros
 
plt.figure(figsize=(18, 10))
plt.plot(profundidade, gradiente_fratura)
plt.xlabel('Profundidade (m)')
plt.ylabel('Gradiente de Fratura (lbf/gal)')
plt.grid(which='both', linestyle='--', linewidth=0.5, axis='both', color='gray')
plt.show()
plt.close()
 

###################
# Plotagem do gráfico da Janela Operacional
# Criação da figura e do eixo
fig, ax = plt.subplots()

plt.figure(figsize=(12, 12))
# Plotagem das curvas
ax.plot(gradiente_sobrecarga[1:], profundidade[1:], label='Gradiente de Sobrecarga', color='blue')
ax.plot(gradiente_pressao_poros, profundidade, label='Gradiente de Poros', color='green')
ax.plot(gradiente_colapso, profundidade, label='Gradiente de Colapso', color='red')
ax.plot(gradiente_fratura, profundidade, label='Gradiente de Fratura', color='black')
# Adicionar a legenda com descrição e cores
ax.legend(loc='best', fontsize='small', title="Significado das Curvas")

# Preenchimento da área entre as curvas gradiente_colapso e gradiente_sobrecarga
ax.fill_betweenx(profundidade, gradiente_colapso, gradiente_fratura, color='yellow', alpha=0.3)

# Inversão do eixo y para que a profundidade cresça para baixo
ax.invert_yaxis()

# Adição dos rótulos dos eixos
ax.set_xlabel('Gradientes (lbf/gal)')
ax.set_ylabel('Profundidade (m)')

# Adição de linhas de grade
ax.grid(True, linestyle='--')

# Exibição da plotagem
plt.show()

plt.close()


######################################
###################
###################
###################
###################
###################
######################################


'''

###################
# Plotagem de Círculo de Mohr

# Definindo a prof de interesse para o círculo de mohr
prof_interesse = 1700
# prof_interesse = float(input("Qual é a profundidade de interesse para geração do círculo de Mohr? "))
print("A profundidade de interesse para geração do círculo de Mohr é:", prof_interesse, ' m')

# cálculo da profundidade da base das camadas
prof_mohr = pd.DataFrame(profundidade)

# Cria uma série deslocada para comparar com a série original
prof_mohr_deslocada = prof_mohr.shift(1)

# Cria a variável 'indices_mohr' que é True apenas para o intervalo que contém a prof_mohr
indices_mohr = (prof_interesse > prof_mohr_deslocada.iloc[:, 0]) & (prof_interesse <= prof_mohr.iloc[:, 0])

# Extraindo os valores para plotagem do círculo de Mohr
tensao_hor_max_interesse = float(tensao_hor_max[indices_mohr])
tensao_hor_min_interesse = float(tensao_hor_min[indices_mohr])
pressao_poros_interesse = float(pressao_poros_estimada[indices_mohr])
tensao_sobrecarga_interesse = float(tensao_sobrecarga[indices_mohr])

# Cálculo da pressão do fluido de perfuração na profundidade de interesse
# gradiente_fluido = float(input("Qual é o gradiente de pressão do fluido de perfuração? "))
gradiente_fluido = 11
print("O gradiente do fluido de perfuração é :", gradiente_fluido, ' lbf/gal')
pressao_fluido_prof_interesse = gradiente_fluido * 0.1704 * prof_interesse
print("A pressão exercida pelo fluido de perfuração na profundidade de:   ", prof_interesse, " m é :",
      pressao_fluido_prof_interesse, ' psi')

# Cálculo das pressões tangenciais máxima e mínima e da pressão radial
tensao_radial = pressao_fluido_prof_interesse - pressao_poros_interesse
tensao_tang_maxima = (3 * tensao_hor_max_interesse) - tensao_hor_min_interesse - pressao_fluido_prof_interesse - pressao_poros_interesse
tensao_tang_minima = (3 * tensao_hor_min_interesse) - tensao_hor_max_interesse - pressao_fluido_prof_interesse - pressao_poros_interesse

# Identificando quais tensões são a mínima e a máxima
tensao_max_circulo = max(tensao_radial, tensao_tang_maxima, tensao_tang_minima)
tensao_min_circulo = min(tensao_radial, tensao_tang_maxima, tensao_tang_minima)
centro_circulo = (tensao_max_circulo + tensao_min_circulo) / 2
raio_circulo = (tensao_max_circulo - tensao_min_circulo) / 2

# Plotando o diagrama de Mohr
# Crie um array de ângulos de 0 a 2pi
angulos = np.linspace(0, 2 * np.pi, 100)

# Calcule as coordenadas x e y do círculo
x = centro_circulo + raio_circulo * np.cos(angulos)
y = raio_circulo * np.sin(angulos)  # a coordenada y do centro é 0

# Definição das propriedades da rocha
# coesao = 5000  # psi
coesao = S0
# atrito_interno = np.tan(np.deg2rad(34))  # graus
atrito_interno = np.tan(phi_rad)
limite_tracao_mohr = -750  # psi

# Crie a plotagem
plt.figure(figsize=(12, 12))
plt.plot(x, y)

# Adicione os eixos x e y
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)

# Adicione a linha vertical a partir da variável 'limite_tracao_mohr'
plt.vlines(x=limite_tracao_mohr, ymin=0, ymax= 1.3 * raio_circulo, colors='r', label='Limite de tração')

# Adicione a reta usando a equação y = x_atrito*atrito_interno + coesao
x_linha2 = np.linspace(0, 1.5 * tensao_max_circulo, 2)
y_linha2 = (x_linha2 * atrito_interno) + coesao
plt.plot(x_linha2, y_linha2, label='Limite de Mohr-Coulomb - colapso da formação por cisalhamento')

plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.show()
'''

