import requests
import json

# URL do seu servidor local
url = 'http://127.0.0.1:5000/prever'

# Dados de uma casa imaginária para testar
dados_casa = {
    'area': 120,
    'quartos': 3,
    # optionally provide these if you know a supported city or imvl_type value
    # 'city': 'Sao Paulo',
    # 'imvl_type': 'apartamentos'
}

# Envia a pergunta para o servidor
print(f"Perguntando ao oráculo o preço de uma casa de {dados_casa['area']}m² com {dados_casa['quartos']} quartos...")
resposta = requests.post(url, json=dados_casa)

# Mostra a resposta divina
print("Resposta recebida:", resposta.json())