import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# --- AQUI ESTÁ A MUDANÇA MÁGICA ---
# Carregando os dados do arquivo CSV que está na mesma pasta
# Certifique-se de que 'casas.csv' está na mesma pasta deste script
try:
    df = pd.read_csv('casas.csv')
    print("Dados carregados com sucesso do pergaminho 'casas.csv'!")
except FileNotFoundError:
    print("ERRO: O arquivo 'casas.csv' não foi encontrado. Verifique se o nome está correto.")
    exit()

# Visualizando as primeiras linhas para garantir que o Python leu corretamente
print(df.head())

# --- O RESTO SEGUE O RITUAL PADRÃO ---

# Definindo quem é pergunta (X) e quem é resposta (y)
# Ajuste os nomes 'area', 'quartos' e 'preco' conforme o cabeçalho do seu CSV
X = df[['area', 'quartos']]
y = df['preco']

# Dividindo em Treino e Teste [cite: 25]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando e treinando o modelo [cite: 27]
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Salvando o cérebro da máquina
joblib.dump(modelo, 'modelo_imoveis.pkl')
print("Modelo treinado e salvo!")