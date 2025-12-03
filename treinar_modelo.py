import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
import numpy as np
import os  # <--- Importante

# --- BLINDAGEM DO CAMINHO DO ARQUIVO ---
# Pega o diretório onde ESTE arquivo (treinar_modelo.py) está
diretorio_base = os.path.dirname(os.path.abspath(__file__))
caminho_csv = os.path.join(diretorio_base, 'dataZAP.csv')

print(f"Procurando arquivo em: {caminho_csv}")

# 1. Carregar o dataset real
try:
    df = pd.read_csv(caminho_csv, sep=';') 
    print("Arquivo carregado com sucesso!")
    print("Colunas encontradas:", df.columns.tolist()) # Debug para ver se os nomes batem
except FileNotFoundError:
    print(f"\nERRO FATAL: O arquivo não foi encontrado no caminho: {caminho_csv}")
    print("Verifique se o nome do arquivo é exatamente 'dataZAP.csv' e se ele está na mesma pasta do script.")
    exit()
except Exception as e:
    print(f"\nERRO INESPERADO ao ler o CSV: {e}")
    exit()

# ... (O resto do código continua igual a partir daqui: passo 2, seleção de colunas, etc)

# 2. Seleção de Colunas Relevantes
# Mapeando as colunas do CSV para nomes mais amigáveis
colunas_map = {
    'listing.usableAreas': 'area',
    'listing.bedrooms': 'quartos',
    'listing.bathrooms': 'banheiros',
    'listing.parkingSpaces': 'vagas',
    'listing.pricingInfo.price': 'preco'
}

# Verificar se as colunas existem
colunas_existentes = [c for c in colunas_map.keys() if c in df.columns]
df_selecionado = df[colunas_existentes].rename(columns=colunas_map)

# 3. Limpeza e Tratamento de Dados
print("Tratando dados...")

# Função para limpar preços (ex: '1.300' -> 1300.0)
def limpar_preco(valor):
    if pd.isna(valor):
        return np.nan
    if isinstance(valor, str):
        valor = valor.replace('.', '').replace(',', '.')
    try:
        return float(valor)
    except ValueError:
        return np.nan

# Função para limpar dados numéricos gerais
def limpar_numero(valor):
    if pd.isna(valor):
        return 0 # Assume 0 se for nulo (ex: sem vaga)
    if isinstance(valor, str):
        # Remove caracteres não numéricos
        valor = ''.join(filter(str.isdigit, valor))
        if not valor: return 0
    return float(valor)

# Aplicar limpeza
df_selecionado['preco'] = df_selecionado['preco'].apply(limpar_preco)
df_selecionado['area'] = df_selecionado['area'].apply(limpar_numero)
df_selecionado['quartos'] = df_selecionado['quartos'].apply(limpar_numero)
df_selecionado['banheiros'] = df_selecionado['banheiros'].apply(limpar_numero)
df_selecionado['vagas'] = df_selecionado['vagas'].apply(limpar_numero)

# Remover linhas que ainda tenham valores nulos ou preço zero
df_selecionado.dropna(inplace=True)
df_selecionado = df_selecionado[df_selecionado['preco'] > 0]

print(f"Dados limpos: {len(df_selecionado)} registros válidos para treinamento.")

# 4. Divisão Treino/Teste
X = df_selecionado[['area', 'quartos', 'banheiros', 'vagas']]
y = df_selecionado['preco']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Treinamento (Regressão Linear - conforme requisitos do PDF)
print("Treinando modelo de Regressão Linear...")
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Avaliação
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Modelo treinado com sucesso!")
print(f"Erro Médio Quadrático (MSE): {mse:.2f}")
print(f"R2 Score (Precisão): {r2:.4f}")

# 7. Salvar o modelo e metadados
# Criar pasta models se não existir
import os
if not os.path.exists('models'):
    os.makedirs('models')

joblib.dump(model, 'models/modelo_imoveis.pkl')

# Salvar a lista de colunas esperadas (input) para a API usar
metadata = {
    "features": list(X.columns),
    "target": "preco",
    "algoritmo": "LinearRegression",
    "r2_score": r2
}

with open('models/modelo_metadata.json', 'w') as f:
    json.dump(metadata, f)

print("Arquivos salvos em 'models/': modelo_imoveis.pkl e modelo_metadata.json")