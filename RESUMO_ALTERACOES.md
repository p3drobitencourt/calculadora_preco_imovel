# Resumo das alterações

Data: 02/12/2025

Resumo curto:
- Adicionei exportação do modelo treinado (joblib) e metadados (colunas/metadata) em `treinar_modelo.py`.
- Atualizei `app.py` para carregar modelo e metadados; agora constrói um vetor de entrada alinhado às colunas do modelo (fallback para versão legada com 2 features quando não houver metadados).
- Criei `verificar_modelo.py` para carregar o modelo e metadados e testar previsões localmente.
- Ajustei `testar_api.py` para indicar campos opcionais (city, imvl_type) quando usando modelo com dummies/one-hot encoding.
- Criei `README.md` com instruções rápidas de instalação e execução e `requirements.txt` com dependências.

Arquivos alterados/criados (detalhe):

- `treinar_modelo.py`
  - Importei `joblib`, `os`, `json` e criei função `export_model(model, feature_columns, version="v0.8")`.
  - Ao finalizar o treinamento, o script executa `export_model(model, list(X.columns))` quando chamado diretamente.
  - `export_model` salva:
    - `models/modelo_imoveis.pkl` (joblib) e também uma cópia em `modelo_imoveis.pkl` na raiz para compatibilidade.
    - `models/modelo_columns.json` e `modelo_columns.json` na raiz contendo a lista ordenada de colunas feature usadas pelo modelo (JSON).
    - `models/modelo_metadata.json` e `modelo_metadata.json` com versão e quantidade de features.
  - Observação: como o dataset foi processado com get_dummies, as colunas do modelo incluem dummies como `listing.address.city_<NomeCidade>` e `imvl_type_<tipo>`.

- `app.py`
  - Adicionei a leitura de `modelo_columns.json` (se existir) e compilação do vector de entrada (`entrada`) alinhado a essa lista de colunas.
  - Permanece compatível com versões antigas do modelo (legado) que esperam apenas [area, quartos] quando `modelo_columns.json` não existe.
  - O servidor também trata mapeamento simples: `area` -> `listing.usableAreas`, `quartos` -> `listing.bedrooms`, campos de `city` e `imvl_type` para setar a dummy correspondente quando válidos.

- `verificar_modelo.py` (novo)
  - Script simples para carregar `modelo_imoveis.pkl` e `modelo_columns.json` e fazer uma previsão de teste offline.
  - Constrói a entrada no mesmo formato que o server: dicionário de colunas preenchendo zeros e setando os valores fornecidos.

- `testar_api.py` (atualizado)
  - Adicionei comentários para campos opcionais `city` e `imvl_type` no payload de teste HTTP, para compatibilidade com o novo modelo.

- `README.md` (novo)
  - Breve descrição do projeto, passos rápidos de setup e execução, comando para treinar, como iniciar o servidor e verificar o modelo.

- `requirements.txt` (novo)
  - Dependências: pandas, numpy, scikit-learn, joblib, flask, flask-cors, requests

- `models/` (pasta gerada)
  - `models/modelo_imoveis.pkl` — joblib do modelo salvo.
  - `models/modelo_columns.json` — lista ordenada de colunas (features) esperadas pelo modelo.
  - `models/modelo_metadata.json` — metadados da versão e número de features.
  - Observação: uma cópia de `modelo_imoveis.pkl`, `modelo_columns.json` e `modelo_metadata.json` também são gravadas na raiz para facilitar carregamento por `app.py` em setups mais simples.

Como usar (atenção rápida):
1. Crie e ative venv e instale dependências:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Treine e exporte o modelo:

```powershell
python treinar_modelo.py
```

Após rodar, os arquivos `modelo_imoveis.pkl`, `modelo_columns.json`, `modelo_metadata.json` e também os arquivos dentro de `models/` serão gerados.

3. Inicie o servidor e teste:

```powershell
python app.py
# Em outro terminal
python testar_api.py
```

4. Teste local offline (opcional):

```powershell
python verificar_modelo.py
```

Observações / Recomendações:
- O `app.py` agora depende de `modelo_columns.json` para alinhar corretamente o vetor de características;
  - Se o `modelo_columns.json` não existir, `app.py` usará comportamento legado (entrada `[area, quartos]`).
- Garanta que os valores passados para `city` e `imvl_type` correspondam exatamente às colunas dummy presentes no `modelo_columns.json` (ex.: `listing.address.city_Sao Paulo` ou `imvl_type_apartamentos`).
- Se desejar um endpoint de deploy, considere usar um WSGI server (gunicorn/uwsgi) e adicionar batch preprocessing e validação de input no server.

Problemas conhecidos e próximos passos sugeridos:
- Ajustar pré-processamento para normalizar/escala (scaling) dos features antes de treinar; isso melhora estabilidade do modelo e previsões que podem parecer anormais.
- Garantir que o processo de exportação registre o (versão/modelo) e as transformações aplicadas (e.g., dummies, normalizações) em `modelo_metadata.json` para audit e replicação.
- Criar um endpoint `/health` e testes automatizados para garantir que a entrada e saída não quebrem quando mudarem as colunas do modelo.

---
Se quiser, posso:
- Adicionar checagens automáticas no `app.py` para validar o payload de entrada e retornar erros amigáveis (e.g., campo `area` obrigatório),
- Criar um script `scripts/atualizar_modelo.sh` (ou PowerShell) para exportar, testar, versionar o modelo automaticamente,
- Implementar pipeline básico de testes unitários para `verificar_modelo.py` e o `app.py`.

Obrigado — diga qual próximo passo prefere que eu implemente.
