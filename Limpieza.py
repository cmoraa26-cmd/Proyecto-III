import pandas as pd
import numpy as np

# ============================================
# CARGA DE DATOS
# ============================================

df = pd.read_csv(
    "conjunto_de_datos_tmodulo_endiseg_2021.csv",
    low_memory=False
)

diccionario = pd.read_csv(
    "diccionario_datos_tmodulo_endiseg_2021.csv",
    low_memory=False
)

# ============================================
# LIMPIEZA DE NOMBRES DE COLUMNAS
# ============================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
)

diccionario.columns = (
    diccionario.columns
    .str.strip()
    .str.lower()
)

# Limpiar espacios en columnas tipo texto
for col in diccionario.columns:
    diccionario[col] = diccionario[col].astype(str).str.strip()

# ============================================
# INFORMACIÓN GENERAL
# ============================================

print('Shape del conjunto de datos')
print(df.shape)

print('\nShape del diccionario')
print(diccionario.shape)

print('\nColumnas del diccionario')
print(diccionario.columns)

print('\nPrimeras filas del diccionario')
print(diccionario.head(20))

# ============================================
# VARIABLES RELACIONADAS CON SUICIDIO
# ============================================

resultado_suicidio = diccionario[
    diccionario.astype(str)
    .apply(
        lambda x: x.str.contains(
            'suic',
            case=False,
            na=False
        )
    )
    .any(axis=1)
]

print('\nVariables relacionadas con suicidio:')
print(resultado_suicidio.T)

# ============================================
# BÚSQUEDA POR PALABRAS CLAVE
# ============================================

palabras_clave = [
    'sexual',
    'género',
    'preferencia',
    'identidad',
    'discrimin',
    'rechazo',
    'agresión',
    'violencia',
    'trato',
    'emocional',
    'depres',
    'triste',
    'ansiedad',
    'familia',
    'pareja',
    'escuela',
    'trabajo'
]

pd.set_option('display.max_colwidth', None)

for palabra in palabras_clave:

    print(f'\n==========={palabra.upper()}=========')

    resultados = diccionario[
        diccionario.astype(str)
        .apply(
            lambda x: x.str.contains(
                palabra,
                case=False,
                na=False
            )
        )
        .any(axis=1)
    ]

    if resultados.empty:
        print('No se encontraron resultados')

    else:
        print(resultados[['nombre_campo', 'nemónico']].head(10))

# ============================================
# VARIABLES IMPORTANTES
# ============================================

Preguntas = [
    'p10_2',
    'p11_4_1',
    'p11_4_2',
    'p11_4_3',
    'p11_4_4',
    'p11_4_5',
    'p12_3_2',
    'p4_15',
    'p4_18_5'
]

resultado_preguntas = diccionario[
    diccionario['nemónico'].isin(Preguntas)
]

print('\nVariables seleccionadas:')
print(resultado_preguntas[['nemónico', 'nombre_campo']])

# ============================================
# VARIABLES PARA EL MODELO
# ============================================

variables_modelo = [

    # Variable objetivo
    'p10_2',

    # Demográficas
    'p4_1',
    'niv',
    'gra',
    'p4_15',

    # Discriminación
    'p8_5',
    'p9_9',
    'p11_6_08',
    'p11_6_11',

    # Violencia
    'p11_4_3',
    'p11_4_5',
    'p11_8_1',

    # Salud mental
    'p10_1_3',
    'p10_1_5',
    'pd4_10_8',

    # Apoyo social
    'p12_4_3',

    # Contexto LGBT+
    'p12_3_1',
    'p12_3_2'
]

# ============================================
# VERIFICAR COLUMNAS EXISTENTES
# ============================================

print('\nVerificando columnas...')

faltantes = [
    col for col in variables_modelo
    if col not in df.columns
]

if len(faltantes) > 0:

    print('\nColumnas NO encontradas:')
    print(faltantes)

else:
    print('\nTodas las columnas fueron encontradas')

# ============================================
# CREAR DATAFRAME DEL MODELO
# ============================================

variables_validas = [
    col for col in variables_modelo
    if col in df.columns
]

modelo_df = df[variables_validas].copy()

print('\nShape del dataframe del modelo')
print(modelo_df.shape)

# ============================================
# VARIABLE OBJETIVO
# ============================================

if 'p10_2' in modelo_df.columns:

    X = modelo_df.drop(columns=['p10_2'])
    y = modelo_df['p10_2']

    print('\nDistribución de clases:')
    print(y.value_counts(dropna=False))

else:

    print('\nNo se encontró la variable objetivo p10_2')

# ============================================
# VALORES FALTANTES
# ============================================

print('\nPorcentaje de valores faltantes:')

na_porcentaje = (
    modelo_df
    .isnull()
    .mean()
    .sort_values(ascending=False)
    * 100
)

print(na_porcentaje)

# ============================================
# TIPOS DE DATOS
# ============================================

print('\nTipos de datos:')
print(modelo_df.dtypes)

# ============================================
# GUARDAR DATAFRAME LIMPIO
# ============================================

modelo_df.to_csv(
    'dataset_modelo_suicidio.csv',
    index=False
)

print('\nDataset limpio guardado correctamente')