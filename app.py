from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import json
import os

# 1. Inicializa a aplicação Flask
app = Flask(__name__)
CORS(app) # Libera o acesso para o seu futuro site (React)

# 2. Carrega o modelo treinado (O Cérebro)
metadata_cols = None
modelo = None

# Define o caminho correto para o modelo
model_path = os.path.join(os.path.dirname(__file__), 'models', 'modelo_imoveis.pkl')
metadata_path = os.path.join(os.path.dirname(__file__), 'models', 'modelo_columns.json')

try:
    modelo = joblib.load(model_path)
    # attempt to load metadata columns file (if available)
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata_cols = json.load(f)

    print("Modelo carregado com sucesso!")
    if metadata_cols is not None:
        print(f"Modelo espera {len(metadata_cols)} features.")
except Exception as e:
    print("ERRO CRÍTICO: O arquivo do modelo não foi encontrado.")
    print(f"Caminho esperado: {model_path}")
    print(e)

# 3. Rota principal (apenas para ver se está vivo)
@app.route('/')
def home():
    return "O Oráculo das Casas está ONLINE!"

# 4. Rota de Previsão (Onde a mágica acontece)
@app.route('/prever', methods=['POST'])
def prever():
    dados = request.get_json()
    
    # CORREÇÃO: Definir as variáveis ANTES do if/else para que existam sempre
    # Usamos .get() e um valor default (0) para segurança
    area = float(dados.get('area', 0))
    quartos = int(dados.get('quartos', 0))

    # Se metadata_cols não existir, usa o modo legado
    if metadata_cols is None:
        # Backwards compatibility: legacy model expecting [area, quartos]
        entrada = [[area, quartos]]
    else:
        # Build a dict with all columns set to 0
        row = {col: 0 for col in metadata_cols}

        # Map common aliased fields using the variables we extracted above
        if 'listing.usableAreas' in row:
            row['listing.usableAreas'] = area
        
        if 'listing.bedrooms' in row:
            row['listing.bedrooms'] = quartos

        # Outros campos opcionais
        if 'bathrooms' in dados:
            if 'listing.bathrooms' in row:
                row['listing.bathrooms'] = float(dados['bathrooms'])
        if 'parkingSpaces' in dados:
            if 'listing.parkingSpaces' in row:
                row['listing.parkingSpaces'] = float(dados['parkingSpaces'])
        
        # Handle city and type columns (one-hot columns)
        if 'city' in dados:
            city_col = f"listing.address.city_{dados['city']}"
            if city_col in row:
                row[city_col] = 1
        if 'imvl_type' in dados:
            type_col = f"imvl_type_{dados['imvl_type']}"
            if type_col in row:
                row[type_col] = 1

        # Build ordered list of values according to metadata_cols
        entrada = [[row[c] for c in metadata_cols]]
    
    # Faz a previsão
    preco_estimado = modelo.predict(entrada)[0]
    
    # Devolve a resposta em formato JSON
    # Agora 'area' e 'quartos' estão garantidamente definidos!
    return jsonify({
        'area': area,
        'quartos': quartos,
        'preco_previsto': round(preco_estimado, 2)
    })

# 5. Inicia o servidor
if __name__ == '__main__':
    app.run(debug=True, port=5000)