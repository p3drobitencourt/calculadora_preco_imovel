# Calculadora Preço Imóvel

This repo trains a rental price model on ZAP data and exposes a simple Flask API to predict rental price.

## Files
- `treinar_modelo.py`: trains the model and exports the model and metadata to `models/` and root files like `modelo_imoveis.pkl` and `modelo_columns.json`.
- `app.py`: a Flask server that loads the model and `modelo_columns.json` to prepare inputs and responds to `/prever` requests.
- `testar_api.py`: simple client script to test the `/prever` endpoint.
- `verificar_modelo.py`: script to test model loading and an offline prediction.
- `dataZAP.csv`: the dataset used for training.

## Quick steps
1. Create a Python venv and install dependencies:

```powershell
python -m venv venv; venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

2. Train and export the model:

```powershell
python treinar_modelo.py
```

This will create `models/modelo_imoveis.pkl`, `modelo_columns.json` and `modelo_metadata.json`.

3. Start the API server:

```powershell
python app.py
```

4. Test with the sample client (or the `verificar_modelo.py`):

```powershell
python testar_api.py
python verificar_modelo.py
```

## Notes
- A `modelo_columns.json` file is generated so the server can construct a consistent input vector for the model (handling hot-encoded columns for city and imovel types).
- If you have an old version of `modelo_imoveis.pkl` that expects 2 features, the server will detect missing `modelo_columns.json` and fallback to legacy behavior (area, quartos only).
