import joblib
import json

# Load model and metadata
try:
    modelo = joblib.load('modelo_imoveis.pkl')
    print('Loaded model modelo_imoveis.pkl')
except Exception as e:
    print('Failed to load model:', e)
    exit(1)

cols = None
try:
    with open('modelo_columns.json', 'r', encoding='utf-8') as f:
        cols = json.load(f)
    print(f'Loaded metadata for {len(cols)} features')
except Exception as e:
    print('Failed to load metadata columns:', e)

# Build a sample input for a prediction
sample = {
    'area': 75,
    'quartos': 2,
    'city': None,  # replace with valid value if you know one
    'imvl_type': None
}

if cols is None:
    # If no metadata, try legacy prediction
    entrada = [[sample['area'], sample['quartos']]]
else:
    row = {c: 0 for c in cols}
    if 'listing.usableAreas' in row:
        row['listing.usableAreas'] = float(sample['area'])
    if 'listing.bedrooms' in row:
        row['listing.bedrooms'] = int(sample['quartos'])

    # If you want to specify a city or type, set the right dummies
    # Example: column names are like 'listing.address.city_Sao Paulo' or 'imvl_type_apartamentos'
    if sample['city']:
        colname = f"listing.address.city_{sample['city']}"
        if colname in row:
            row[colname] = 1
    if sample['imvl_type']:
        tcol = f"imvl_type_{sample['imvl_type']}"
        if tcol in row:
            row[tcol] = 1

    entrada = [[row[c] for c in cols]]

# Make prediction
try:
    pred = modelo.predict(entrada)[0]
    print('Prediction result:', pred)
except Exception as e:
    print('Prediction failed:', e)

print('Done')
