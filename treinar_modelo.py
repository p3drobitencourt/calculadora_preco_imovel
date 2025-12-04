import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. Carregamento
df = pd.read_csv('dataZAP.csv', sep=';')

# 2. Filtro de Negócio (Aluguel)
df_clean = df[df['listing.pricingInfo.isRent'] == True].copy()

# 3. Filtro de Escopo (Restrito)
# Apenas os 3 principais tipos residenciais
tipos_permitidos = ['apartamentos', 'casas', 'casas-de-condominio']
df_clean = df_clean[df_clean['imvl_type'].isin(tipos_permitidos)]

# 4. Features e Target
features = [
    'listing.usableAreas',    
    'listing.bedrooms',       
    'listing.bathrooms',      
    'listing.parkingSpaces',  
    'listing.address.city',   
    'imvl_type'               
]
target = 'listing.pricingInfo.rentalPrice'

df_model = df_clean[features + [target]].copy()

# 5. Conversão Numérica
cols_numeric = ['listing.usableAreas', 'listing.bathrooms', 'listing.parkingSpaces', target]
for col in cols_numeric:
    df_model[col] = pd.to_numeric(df_model[col], errors='coerce')

df_model.dropna(inplace=True)

# 6. Filtro de Estabilidade (Cidades com volume relevante)
counts = df_model['listing.address.city'].value_counts()
cidades_validas = counts[counts >= 20].index
df_model = df_model[df_model['listing.address.city'].isin(cidades_validas)]

# 7. Remoção de Outliers (IQR)
cols_outlier = ['listing.usableAreas', target]
for col in cols_outlier:
    Q1 = df_model[col].quantile(0.25)
    Q3 = df_model[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df_model = df_model[(df_model[col] >= lower) & (df_model[col] <= upper)]

# Garante área mínima coerente para esses tipos (ex: > 10m2)
df_model = df_model[df_model['listing.usableAreas'] > 10]

# 8. Encoding e Treino
df_model = pd.get_dummies(df_model, columns=['listing.address.city', 'imvl_type'], drop_first=True)

X = df_model.drop(target, axis=1)
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# 9. Resultados
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

cidades_disponiveis = sorted(df['listing.address.city'].unique().tolist())
print(cidades_disponiveis)

print(f"--- Release Candidate V0.8 ---")
print(f"R²: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")


def export_model(model, feature_columns, version="v0.8", model_name='modelo_imoveis.pkl'):
    """Export the trained model and save metadata (columns) so it can be loaded later.

    Saves:
    - model file (joblib) as `models/modelo_imoveis.pkl`
    - columns json as `models/modelo_columns.json`
    - a simple metadata json with shape and version info as `models/modelo_metadata.json`
    """
    print("Exporting model...")
    os.makedirs('models', exist_ok=True)

    # Save only in models/ folder
    model_path = os.path.join('models', model_name)
    columns_path = os.path.join('models', 'modelo_columns.json')
    metadata_path = os.path.join('models', 'modelo_metadata.json')

    joblib.dump(model, model_path)
    print(f"✓ Model saved to: {model_path}")

    with open(columns_path, 'w', encoding='utf-8') as f:
        json.dump(feature_columns, f, ensure_ascii=False, indent=2)
    print(f"✓ Columns saved to: {columns_path}")

    metadata = {
        'version': version,
        'n_features': len(feature_columns),
        'feature_columns': feature_columns
    }
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"✓ Metadata saved to: {metadata_path}")


if __name__ == '__main__':
    # Export model with metadata for serving
    export_model(model, list(X.columns))