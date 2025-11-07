import gzip
import json
import pickle
import zipfile
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def load_data_from_zip(zip_path, csv_name):
    """
    Carga un archivo CSV desde un archivo ZIP.
    
    Args:
        zip_path: Ruta al archivo ZIP
        csv_name: Nombre del CSV dentro del ZIP
    
    Returns:
        DataFrame con los datos cargados
    """
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(csv_name) as f:
            df = pd.read_csv(f)
    # Eliminar columnas de índice no deseadas
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    return df


def preprocess_data(df, year_reference=2021):
    """
    Preprocesa los datos del dataset de vehículos.
    
    - Crea la columna Age a partir de Year
    - Elimina columnas innecesarias
    - Aplica transformación logarítmica a Selling_Price
    
    Args:
        df: DataFrame con los datos originales
        year_reference: Año de referencia para calcular Age
    
    Returns:
        DataFrame preprocesado
    """
    df = df.copy()
    df['Age'] = year_reference - df['Year']
    df = df.drop(['Year', 'Car_Name'], axis=1)
    df = df.dropna()
    
    # Transformación logarítmica para reducir el impacto de outliers
    df['Selling_Price'] = np.log1p(df['Selling_Price'])
    
    return df


def create_pipeline():
    """
    Crea el pipeline de preprocesamiento y modelado.
    
    El pipeline incluye:
    1. One-hot encoding para variables categóricas
    2. MinMaxScaler para variables numéricas
    3. SelectKBest para selección de características
    4. LinearRegression como modelo final
    
    Returns:
        Pipeline configurado
    """
    # Definir columnas por tipo
    categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission']
    numerical_features = ['Driven_kms', 'Owner', 'Age']
    passthrough_features = ['Selling_Price']
    
    # Preprocesador con ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', MinMaxScaler(), numerical_features),
            ('pass', 'passthrough', passthrough_features)
        ],
        remainder='drop'
    )
    
    # Pipeline completo
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selector', SelectKBest(score_func=f_regression)),
        ('regressor', LinearRegression())
    ])
    
    return pipeline


def train_model(X_train, y_train):
    """
    Entrena el modelo usando GridSearchCV para optimizar hiperparámetros.
    
    Args:
        X_train: Características de entrenamiento
        y_train: Variable objetivo de entrenamiento
    
    Returns:
        Modelo entrenado (GridSearchCV)
    """
    pipeline = create_pipeline()
    
    # Definir grid de hiperparámetros
    param_grid = {
        'selector__k': list(range(4, 12))
    }
    
    # GridSearchCV con validación cruzada
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Mejores parámetros: {grid_search.best_params_}")
    print(f"Mejor score CV: {grid_search.best_score_:.4f}")
    
    return grid_search


def calculate_metrics(y_true, y_pred, dataset_name):
    """
    Calcula métricas de evaluación del modelo.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        dataset_name: Nombre del dataset ('train' o 'test')
    
    Returns:
        Diccionario con las métricas calculadas
    """
    return {
        'type': 'metrics',
        'dataset': dataset_name,
        'r2': float(r2_score(y_true, y_pred)),
        'mse': float(mean_squared_error(y_true, y_pred)),
        'mad': float(mean_absolute_error(y_true, y_pred))
    }


def save_model(model, filepath):
    """
    Guarda el modelo entrenado en formato comprimido.
    
    Args:
        model: Modelo entrenado
        filepath: Ruta donde guardar el modelo
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with gzip.open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Modelo guardado en: {filepath}")


def save_metrics(metrics_list, filepath):
    """
    Guarda las métricas en formato JSON Lines.
    
    Args:
        metrics_list: Lista de diccionarios con métricas
        filepath: Ruta donde guardar las métricas
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for metrics in metrics_list:
            f.write(json.dumps(metrics) + '\n')
    print(f"Métricas guardadas en: {filepath}")


def main():
    """
    Función principal que ejecuta todo el pipeline de entrenamiento.
    """
    # Cargar datos
    print("Cargando datos...")
    train_df = load_data_from_zip('files/input/train_data.csv.zip', 'train_data.csv')
    test_df = load_data_from_zip('files/input/test_data.csv.zip', 'test_data.csv')
    
    # Preprocesar datos
    print("Preprocesando datos...")
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)
    
    # Dividir en características y variable objetivo
    X_train = train_df.drop('Present_Price', axis=1)
    y_train = np.log1p(train_df['Present_Price'])
    X_test = test_df.drop('Present_Price', axis=1)
    y_test = np.log1p(test_df['Present_Price'])
    
    # Entrenar modelo
    print("Entrenando modelo...")
    model = train_model(X_train, y_train)
    
    # Guardar modelo
    save_model(model, 'files/models/model.pkl.gz')
    
    # Hacer predicciones
    print("Calculando predicciones...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Revertir transformación logarítmica
    y_train_true = np.expm1(y_train)
    y_train_pred = np.expm1(y_train_pred)
    y_test_true = np.expm1(y_test)
    y_test_pred = np.expm1(y_test_pred)
    
    # Calcular métricas
    print("Calculando métricas...")
    train_metrics = calculate_metrics(y_train_true, y_train_pred, 'train')
    test_metrics = calculate_metrics(y_test_true, y_test_pred, 'test')
    
    # Guardar métricas
    save_metrics([train_metrics, test_metrics], 'files/output/metrics.json')
    
    # Mostrar resultados
    print("\n" + "="*50)
    print("RESULTADOS FINALES")
    print("="*50)
    print(f"Train - R2: {train_metrics['r2']:.3f}, MSE: {train_metrics['mse']:.3f}, MAD: {train_metrics['mad']:.3f}")
    print(f"Test  - R2: {test_metrics['r2']:.3f}, MSE: {test_metrics['mse']:.3f}, MAD: {test_metrics['mad']:.3f}")
    print("="*50)
    print("\nProceso completado")


if __name__ == '__main__':
    main()