#REQ 1
# faça os imports que julgar necessários
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

#REQ 2
#essa função deve devolver a base de dados
def ler_base():
  dataset = pd.read_csv(r'dados.csv')

  #REQ 3
#essa função recebe a base lida anteriormente
#ela deve devolver uma tupla contendo as features e a classe
def dividir_em_features_e_classe(base):
  x = base.iloc[:, :-1]
  y = base.iloc[:, -1]
  return x, y

#REQ 4
#essa função recebe as features
#ela deve devolver as features da seguinte forma
#Valores faltantes da coluna "Gastos com pesquisa e desenvolvimento": substituir pela média
#Valores faltantes da coluna "Gastos com administracao": substituir pela mediana
#Valores faltantes da coluna "Gastos com marketing": Substituir por zero
#Valores faltantes da coluna "Estado": Substituir pela moda
def lidar_com_valores_faltantes(features):

  features['Gastos com pesquisa e desenvolvimento'].fillna(features['Gastos com pesquisa e desenvolvimento'].mean(), inplace=True)

  features['Gastos com administracao'].fillna(features['Gastos com administracao'].median(), inplace=True)

  features['Gastos com marketing'].fillna(0, inplace=True)

  features['Estado'].fillna(features['Estado'].mode()[0], inplace=True)
  
  return features

#REQ 5
#essa função recebe as features
#ela deve devolver as features da seguinte forma
#Variável "Estado": Codificar com OneHotEncoding
def codificar_categoricas(features):

  ct = ColumnTransformer(
      [('encoder', OneHotEncoder(drop='first'), ['Estado'])],
      remainder='passthrough'
  )
  
  features_encoded = ct.fit_transform(features)
  
  return features_encoded

#REQ 6
#essa função recebe as features e a classe
#ela deve devolver uma tupla com 4 itens
# features de treinamento, features de teste, classe de treinamento, classe de teste
# a base de treinamento deve ter 75% das instâncias
def obter_bases_de_treinamento_e_teste(features, classe):

  y_array = classe.values
  
  x_train, x_test, y_train, y_test = train_test_split(
      features, y_array, test_size=0.25, random_state=42)
  return x_train, x_test, y_train, y_test

#REQ 7
#essa função recebe as features de treinamento e de teste
#ela deve devolver uma tupla com 2 itens, da seguinte forma
#todas as variáveis normalizadas com o método MinMax
def normalizar_features(features_treinamento, features_teste):
  scaler = StandardScaler()
  
  features_treinamento_scaled = scaler.fit_transform(features_treinamento)
  
  features_teste_scaled = scaler.transform(features_teste)
  
  return features_treinamento_scaled, features_teste_scaled

#REQ 8
def vai():  
  # le a base
  base_de_dados = ler_base()
  
  # divide features e classes
  features, classe = dividir_em_features_e_classe(base_de_dados)
  
  # lida com valores faltantes
  features_tratadas = lidar_com_valores_faltantes(features)
  
  # codifica as variaveis
  features_codificadas = codificar_categoricas(features_tratadas)
  
  # obtem bases de treino e teste
  x_train, x_test, y_train, y_test = obter_bases_de_treinamento_e_teste(features_codificadas, classe)
  
  # normaliza base
  x_train_final, x_test_final = normalizar_features(x_train, x_test)
  

  ###### exibe as 4 bases
  print("="*30)
  print("RESULTADOS DO PRÉ-PROCESSAMENTO")
  print("="*30)
  
  print(f"\nShape de X_train (features de treino normalizadas): {x_train_final.shape}")
  print(x_train_final)
  
  print(f"\nShape de X_test (features de teste normalizadas): {x_test_final.shape}")
  print(x_test_final)

  print(f"\nShape de y_train (classe de treino): {y_train.shape}")
  print(y_train)

  print(f"\nShape de y_test (classe de teste): {y_test.shape}")
  print(y_test)
  print("\nPré-processamento concluído.")
  print("="*30)

vai()