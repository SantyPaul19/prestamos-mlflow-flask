import pandas as pd
import sqlite3

# Cargar CSV
df = pd.read_csv('..//data//train.csv')

# Inspecciona columnas y tipos
print(df.head())
print(df.info())

# Opcional: limpiar datos (ejemplo simple)
df = df.dropna()

# Guardar en SQLite
conn = sqlite3.connect('prestamos.db')
df.to_sql('prestamos', conn, if_exists='replace', index=False)
conn.close()

print("Datos cargados en SQLite exitosamente.")
