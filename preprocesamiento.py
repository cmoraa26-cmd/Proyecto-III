import pandas as pd
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def cargar_datos():
    df = pd.read_csv(
        'dataset_modelo_suicidio.csv'
    )
    
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )
    df = df.dropna(subset=['p10_2'])
    
    X = df.drop(columns=['p10_2'])
    y = df['p10_2']
    
    imputer = SimpleImputer(
        strategy='most_frequent'
    )
    
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed,
        y,
        test_size=0.2,
        stratify=y
    )
    
    print(f"Porcentaje clase minoritaria: {y_train.value_counts(normalize=True)[1]*100:.2f}%")
    
    smote = SMOTE(random_state=42)
    
    X_train_bal, y_train_bal = smote.fit_resample(
    X_train,
    y_train
    )
    
    return (
        X_train_bal,
        X_test,
        y_train_bal,
        y_test
    )