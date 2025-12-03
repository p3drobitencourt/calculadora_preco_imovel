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
colunas_modelo = []
info_modelo = {}

def carregar_inteligencia():
    """Função para carregar o modelo e evitar falhas na inicialização"""
    global modelo, colunas_modelo, info_modelo
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
            modelo = joblib.load(MODEL_PATH)
            with open(META_PATH, 'r') as f:
                meta = json.load(f)
                colunas_modelo = meta.get("features", [])
                info_modelo = meta.get("performance", {})
            print(f"✅ Modelo carregado com sucesso! (R²: {info_modelo.get('r2_score', 'N/A'):.2f})")
        else:
            print("⚠️ AVISO: Arquivos do modelo não encontrados. Rode o treinar_modelo.py primeiro.")
    except Exception as e:
        print(f"❌ Erro fatal ao carregar modelo: {e}")

# Carrega ao iniciar
carregar_inteligencia()

@app.route('/')
def home():
    """Rota de verificação de saúde da API"""
    status = "Ativo" if modelo else "Inativo (Modelo não carregado)"
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