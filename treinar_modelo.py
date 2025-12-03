import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
import numpy as np
import os

# --- 1. CONFIGURAÇÃO E CARREGAMENTO ---
diretorio_base = os.path.dirname(os.path.abspath(__file__))
caminho_csv = os.path.join(diretorio_base, 'dataZAP.csv')

print(f"--- INICIANDO TREINAMENTO OTIMIZADO ---")
try:
    df = pd.read_csv(caminho_csv, sep=';') 
except FileNotFoundError:
    print("ERRO: Arquivo dataZAP.csv não encontrado.")
    exit()

# --- 2. SELEÇÃO E LIMPEZA BÁSICA ---
colunas_map = {
    'listing.usableAreas': 'area',
    'listing.bedrooms': 'quartos',
    'listing.bathrooms': 'banheiros',
    'listing.parkingSpaces': 'vagas',
    'listing.pricingInfo.price': 'preco'
}

# Filtra colunas e renomeia
df_selecionado = df[[c for c in colunas_map.keys() if c in df.columns]].rename(columns=colunas_map)

# Função para converter textos em números
def limpar_valor(valor):
    if pd.isna(valor): return np.nan
    if isinstance(valor, str):
        valor = valor.replace('.', '').replace(',', '.')
        valor = ''.join(filter(lambda x: x.isdigit() or x == '.', valor))
    try:
        val = float(valor)
        return val if val > 0 else np.nan
    except:
        return np.nan

# Aplica a limpeza
for col in df_selecionado.columns:
    df_selecionado[col] = df_selecionado[col].apply(limpar_valor)

df_selecionado.dropna(inplace=True)
qtd_inicial = len(df_selecionado)

# --- 3. REFINAMENTO ESTATÍSTICO (O SEGREDO DA MELHORA) ---
# Para a Regressão Linear funcionar bem, removemos os extremos (outliers).
# Vamos manter apenas o "miolo" do mercado (entre os 10% e 90% das faixas de preço e área)

# Definindo limites aceitáveis (Regra de Pareto adaptada)
min_preco = df_selecionado['preco'].quantile(0.10) # Remove os 10% mais baratos
max_preco = df_selecionado['preco'].quantile(0.90) # Remove os 10% mais caros (luxo/erros)

min_area = df_selecionado['area'].quantile(0.10)   # Remove kitnets minúsculas ou erros
max_area = df_selecionado['area'].quantile(0.90)   # Remove galpões ou mansões gigantes

# Filtragem Agressiva
df_filtrado = df_selecionado[
    (df_selecionado['preco'] >= min_preco) & 
    (df_selecionado['preco'] <= max_preco) &
    (df_selecionado['area'] >= min_area) & 
    (df_selecionado['area'] <= max_area)
]

qtd_final = len(df_filtrado)
print(f"Dados filtrados: de {qtd_inicial} para {qtd_final} registros.")
print(f"Considerando imóveis entre R$ {min_preco:,.0f} e R$ {max_preco:,.0f}")
print(f"Considerando áreas entre {min_area:.0f}m² e {max_area:.0f}m²")

# --- 4. TREINAMENTO (Linear Regression) ---
X = df_filtrado[['area', 'quartos', 'banheiros', 'vagas']]
y = df_filtrado['preco']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# --- 5. AVALIAÇÃO ---
predictions = model.predict(X_test)
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print(f"\n--- PERFORMANCE DO MODELO ---")
print(f"R² Score (Precisão): {r2:.4f}")
print(f"Erro Médio Absoluto: R$ {mae:,.2f}")

# --- 6. SALVAR ---
if not os.path.exists('models'):
    os.makedirs('models')

joblib.dump(model, 'models/modelo_imoveis.pkl')

# Salva metadata para a API usar
metadata = {
    "features": list(X.columns),
    "target": "preco",
    "algoritmo": "LinearRegression",
    "r2_score": r2,
    "mae": mae,
    "faixa_preco_treino": [min_preco, max_preco] # Informação útil
}

with open('models/modelo_metadata.json', 'w') as f:
    json.dump(metadata, f)

print("\nModelo atualizado e salvo com sucesso!")