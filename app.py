from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# 1. Inicializa a aplicação Flask
app = Flask(__name__)
CORS(app) # Libera o acesso para o seu futuro site (React)

# 2. Carrega o modelo treinado (O Cérebro)
# O arquivo 'modelo_imoveis.pkl' DEVE estar na mesma pasta
try:
    modelo = joblib.load('modelo_imoveis.pkl')
    print("Modelo carregado com sucesso!")
except:
    print("ERRO CRÍTICO: O arquivo 'modelo_imoveis.pkl' não foi encontrado.")

# 3. Rota principal (apenas para ver se está vivo)
@app.route('/')
def home():
    return "O Oráculo das Casas está ONLINE!"

# 4. Rota de Previsão (Onde a mágica acontece)
@app.route('/prever', methods=['POST'])
def prever():
    dados = request.get_json()
    
    # Pega os dados enviados (Area e Quartos)
    area = float(dados['area'])
    quartos = int(dados['quartos'])
    
    # Prepara para o modelo (o modelo espera uma lista de listas)
    entrada = [[area, quartos]]
    
    # Faz a previsão
    preco_estimado = modelo.predict(entrada)[0]
    
    # Devolve a resposta em formato JSON
    return jsonify({
        'area': area,
        'quartos': quartos,
        'preco_previsto': round(preco_estimado, 2)
    })

# 5. Inicia o servidor
if __name__ == '__main__':
    app.run(debug=True, port=5000)