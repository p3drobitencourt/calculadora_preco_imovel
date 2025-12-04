import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
import numpy as np
import os
import time

# --- 1. CONFIGURAÇÃO DE AMBIENTE BLINDADO ---
# Garante que os caminhos funcionem em qualquer computador ou servidor
diretorio_base = os.path.dirname(os.path.abspath(__file__))
caminho_csv = os.path.join(diretorio_base, 'dataZAP.csv')
caminho_models = os.path.join(diretorio_base, 'models')

# Cria pasta de modelos se não existir
if not os.path.exists(caminho_models):
    os.makedirs(caminho_models)

print(f"--- INICIANDO TREINAMENTO COMPLETO (MODO ALUGUEL) ---")
print(f"Diretório base: {diretorio_base}")

# --- 2. LEITURA BRUTA DOS DADOS ---
# Lemos com dtype=str para evitar que o Python confunda "1.300" (mil e trezentos) com "1.3" (um vírgula três)
try:
    print("Carregando arquivo CSV...")
    df = pd.read_csv(caminho_csv, sep=';', dtype=str, keep_default_na=False)
    print(f"Total de linhas brutas: {len(df)}")
except FileNotFoundError:
    print("ERRO CRÍTICO: O arquivo 'dataZAP.csv' não foi encontrado.")
    exit()
except Exception as e:
    print(f"ERRO DE LEITURA: {e}")
    exit()

# --- 3. MAPEAMENTO E SELEÇÃO DE COLUNAS ---
colunas_map = {
    'listing.usableAreas': 'area',
    'listing.bedrooms': 'quartos',
    'listing.bathrooms': 'banheiros',
    'listing.parkingSpaces': 'vagas',
    'listing.pricingInfo.price': 'preco'
}

# Filtra apenas colunas existentes e renomeia
cols_existentes = [c for c in colunas_map.keys() if c in df.columns]
df = df[cols_existentes].rename(columns=colunas_map)

# --- 4. MOTOR DE LIMPEZA DE DADOS (CLEANING ENGINE) ---
def limpar_valor_monetario(valor_str):
    """
    Converte strings brasileiras complexas para float.
    Exemplos que ele resolve:
    'R$ 1.500,00' -> 1500.0
    '1.200'       -> 1200.0
    '3500'        -> 3500.0
    """
    if pd.isna(valor_str) or valor_str == '': return np.nan
    
    # Remove R$ e espaços extras
    s = str(valor_str).strip().replace('R$', '').strip()
    
    try:
        # Se tiver vírgula, assume que é separador decimal (centavos)
        # Primeiro removemos os pontos de milhar (1.500 -> 1500)
        s = s.replace('.', '')
        # Depois trocamos a vírgula por ponto (1500,50 -> 1500.50)
        s = s.replace(',', '.')
        
        val = float(s)
        return val
    except:
        return np.nan

def limpar_numero_simples(valor):
    """Limpa quartos, vagas e áreas"""
    try:
        # Pega apenas números e pontos
        nums = ''.join(filter(lambda x: x.isdigit() or x == '.', str(valor)))
        if not nums: return 0.0
        return float(nums)
    except:
        return 0.0

print("Higienizando dados numéricos...")
# Aplica a limpeza
df['preco'] = df['preco'].apply(limpar_valor_monetario)
for col in ['area', 'quartos', 'banheiros', 'vagas']:
    if col in df.columns:
        df[col] = df[col].apply(limpar_numero_simples)

# Remove linhas que falharam na conversão (NaN)
df.dropna(inplace=True)

# --- 5. FILTRO DE MERCADO (ALUGUEL) ---
# Aqui definimos a regra de negócio: O que é um aluguel válido?
# Mínimo: R$ 300 (abaixo disso é erro ou vaga de garagem solta)
# Máximo: R$ 50.000 (acima disso é venda ou aluguel industrial gigante)
# Área: entre 10m² e 1000m²

df_final = df[
    (df['preco'] >= 300) & 
    (df['preco'] <= 50000) & 
    (df['area'] >= 10) & 
    (df['area'] <= 1000)
].copy()

qtd = len(df_final)
print(f"Registros válidos para treinamento: {qtd}")

if qtd < 100:
    print("ALERTA: Poucos dados. O modelo pode não ficar preciso.")
else:
    print("Dataset robusto identificado. Prosseguindo...")

# --- 6. TREINAMENTO DA INTELIGÊNCIA ARTIFICIAL ---
print("Treinando modelo de Regressão Linear...")

X = df_final[['area', 'quartos', 'banheiros', 'vagas']]
y = df_final['preco']

# Divisão Treino (80%) / Teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# --- 7. AVALIAÇÃO DE PERFORMANCE ---
predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print(f"\n--- RELATÓRIO DE PERFORMANCE ---")
print(f"Precisão (R² Score): {r2:.4f} (Quanto mais próximo de 1.0, melhor)")
print(f"Erro Médio Absoluto: R$ {mae:,.2f}")

# --- 8. SALVAMENTO E PERSISTÊNCIA ---
print("\nSalvando arquivos para deploy...")

# Salva o modelo binário (.pkl)
joblib.dump(model, os.path.join(caminho_models, 'modelo_imoveis.pkl'))

    Saves:
    - model file (joblib) as `models/modelo_imoveis.pkl`
    - columns json as `models/modelo_columns.json`
    - a simple metadata json with shape and version info as `models/modelo_metadata.json`
    """
    print("Exporting model...")
    os.makedirs('models', exist_ok=True)

    # Save only in models/ folder
    model_path = os.path.join('models', model_name)
    columns_path = os.path.join('models', 'modelo_columns.json')
    metadata_path = os.path.join('models', 'modelo_metadata.json')

    joblib.dump(model, model_path)
    print(f"✓ Model saved to: {model_path}")

    with open(columns_path, 'w', encoding='utf-8') as f:
        json.dump(feature_columns, f, ensure_ascii=False, indent=2)
    print(f"✓ Columns saved to: {columns_path}")

    metadata = {
        'version': version,
        'n_features': len(feature_columns),
        'feature_columns': feature_columns
    }
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"✓ Metadata saved to: {metadata_path}")

print(f"SUCESSO: Modelo treinado e salvo em '{caminho_models}'.")
print("Próximo passo: Execute 'python app.py' para testar a API.")