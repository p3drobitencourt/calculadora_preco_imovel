from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import json
import os

app = Flask(__name__)
CORS(app)  # Permite que o Front-end (React) acesse a API

# Carregar modelo e metadados
MODEL_PATH = 'models/modelo_imoveis.pkl'
META_PATH = 'models/modelo_metadata.json'

modelo = None
colunas_modelo = []

if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
    modelo = joblib.load(MODEL_PATH)
    with open(META_PATH, 'r') as f:
        meta = json.load(f)
        colunas_modelo = meta.get("features", [])
    print("Modelo e metadados carregados com sucesso.")
else:
    print("AVISO: Modelo não encontrado. Execute treinar_modelo.py primeiro.")

@app.route('/')
def home():
    return "API de Previsão de Imóveis Ativa! Use o endpoint /predict para fazer previsões."

@app.route('/predict', methods=['POST'])
def predict():
    if not modelo:
        return jsonify({'error': 'Modelo não disponível. Treine o modelo primeiro.'}), 500

    try:
        dados = request.get_json()
        
        # Validar se todos os campos necessários estão presentes
        input_data = []
        for col in colunas_modelo:
            val = dados.get(col)
            if val is None:
                return jsonify({'error': f'Campo obrigatório ausente: {col}'}), 400
            input_data.append(float(val))

        # Criar DataFrame para previsão (mantém o nome das colunas)
        df_input = pd.DataFrame([input_data], columns=colunas_modelo)
        
        # Realizar previsão
        preco_estimado = modelo.predict(df_input)[0]

        return jsonify({
            'preco_estimado': round(preco_estimado, 2),
            'mensagem': 'Previsão realizada com sucesso!'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Porta 5000 é padrão para desenvolvimento local
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)