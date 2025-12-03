import requests
import json

# Dados de um imóvel simulado (Apartamento padrão: 70m², 2 quartos...)
dados_imovel = {
    "area": 70,
    "quartos": 2,
    "banheiros": 1,
    "vagas": 1
}

url = "http://127.0.0.1:5000/predict"

try:
    print(f"Enviando dados para a API: {dados_imovel}")
    response = requests.post(url, json=dados_imovel)
    
    if response.status_code == 200:
        resultado = response.json()
        print("\n--- SUCESSO DIVINO! ---")
        print(f"Preço Estimado pela IA: R$ {resultado['preco_estimado']:,.2f}")
    else:
        print(f"\nErro na API: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"\nErro de conexão: {e}")
    print("Verifique se o arquivo app.py está rodando no outro terminal!")