import requests
import json

url = "http://127.0.0.1:5000/predict"

# Exemplo: Apartamento padrão de 2 quartos
payload = {
    "area": 60,
    "quartos": 2,
    "banheiros": 1,
    "vagas": 1
}

print("--- TESTE DE PREVISÃO DE ALUGUEL ---")
print(f"Enviando: {payload}")

try:
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print("\n✅ SUCESSO!")
        print(f"Preço Sugerido: {data['preco_formatado']}")
        print(f"Valor Bruto: {data['preco_estimado']}")
    else:
        print(f"\n❌ ERRO NA API ({response.status_code}):")
        print(response.text)
except Exception as e:
    print(f"\n⚠️ FALHA DE CONEXÃO: {e}")
    print("O app.py está rodando em outro terminal?")