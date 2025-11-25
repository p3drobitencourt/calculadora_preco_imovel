import pandas as pd
import numpy as np

# Configuração divina para gerar dados realistas
np.random.seed(42) # Garante que os dados sejam sempre os mesmos se rodar de novo
quantidade = 150   # Mais de 100, conforme a regra do trabalho

# Gerando características aleatórias
areas = np.random.randint(50, 400, quantidade) # Casas de 50m² a 400m²
quartos = np.random.randint(1, 6, quantidade)  # 1 a 5 quartos

# Fórmula Mágica: Preço Base + (Area * Valor m²) + (Quartos * Valor Quarto) + Variação
# Isso garante que existe uma lógica para a Regressão Linear encontrar
precos = 50000 + (areas * 3500) + (quartos * 25000) + np.random.randint(-30000, 30000, quantidade)

# Criando a Tabela
df = pd.DataFrame({
    'area': areas,
    'quartos': quartos,
    'preco': precos
})

# Salvando no arquivo físico
df.to_csv('casas.csv', index=False)
print(f"Sucesso, ó Mestre! O arquivo 'casas.csv' foi gerado com {quantidade} linhas.")