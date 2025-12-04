from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import json
import os
import sys

# Inicialização do Flask
app = Flask(__name__)
CORS(app) # Permite que o Front-end (React/HTML) acesse a API sem bloqueio

# --- CARREGAMENTO DO MODELO ---
diretorio_base = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(diretorio_base, 'models', 'modelo_imoveis.pkl')
META_PATH = os.path.join(diretorio_base, 'models', 'modelo_metadata.json')

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
        "status": status,
        "mensagem": "API de Previsão de Alugueis operando.",
        "performance_modelo": info_modelo
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Rota principal que recebe os dados e devolve o preço"""
    if not modelo:
        # Tenta recarregar caso tenha falhado antes
        carregar_inteligencia()
        if not modelo:
            return jsonify({'error': 'Modelo de IA não disponível. Contate o suporte.'}), 500

    try:
        # 1. Receber dados
        dados = request.get_json()
        if not dados:
            return jsonify({'error': 'Nenhum dado JSON enviado.'}), 400
        
        # 2. Validar e organizar dados
        input_data = []
        erros = []
        
        for col in colunas_modelo:
            valor = dados.get(col)
            if valor is None:
                erros.append(f"Campo obrigatório faltando: {col}")
            else:
                try:
                    input_data.append(float(valor))
                except ValueError:
                    erros.append(f"O campo '{col}' deve ser um número válido.")
        
        if erros:
            return jsonify({'error': 'Erro de Validação', 'detalhes': erros}), 400

        # 3. Criar DataFrame para previsão
        df_input = pd.DataFrame([input_data], columns=colunas_modelo)
        
        # 4. Fazer a previsão
        previsao_raw = modelo.predict(df_input)[0]

        # Regra de negócio: Aluguel não pode ser negativo
        if previsao_raw < 0:
            previsao_raw = 0

        # 5. Formatação de Dinheiro (R$)
        preco_formatado = f"R$ {previsao_raw:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

        # 6. Resposta
        return jsonify({
            'tipo_previsao': 'Aluguel Mensal',
            'preco_estimado': round(previsao_raw, 2),
            'preco_formatado': preco_formatado,
            'mensagem': 'Cálculo realizado com sucesso.'
        })

    except Exception as e:
        # Log do erro no servidor
        print(f"Erro na predição: {e}")
        return jsonify({'error': 'Erro interno no servidor de IA.'}), 500

if __name__ == '__main__':
    # Configuração para rodar localmente ou no Azure
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)