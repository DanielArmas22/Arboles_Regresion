# Importamos las librerías necesarias
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Configuración de la página en Streamlit
st.set_page_config(page_title="Predicción de Edad en el Titanic (Arbol de Regresión)", layout="wide")
st.title("Predicción de Edad en el Titanic con Random Forest Regressor")

# Cargar los datos
@st.cache_data
def load_data():
    # url = 'https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv'
    df = pd.read_csv('titanic.csv')
    return df

# Procesar los datos
@st.cache_data
def preprocess_data(df):
    df = df.drop(['Name', 'Fare'], axis=1)
    df.columns = ['sobrevive', 'clase_social', 'sexo', 'años', 'esposos_hermanos', 'padres_hijos']
    df = pd.get_dummies(df, columns=['sexo'], drop_first=True)
    df.rename(columns={'sexo_masculino': 'sexo_male'}, inplace=True)
    df.dropna(subset=['años'], inplace=True)
    return df

# Cargar y procesar los datos
titanic = load_data()
titanic = preprocess_data(titanic)

# Mostrar los primeros registros de los datos
st.subheader("Datos del Titanic")
st.write(titanic.head())

# Separar en X e y para regresión (predecir 'años')
X = titanic.drop(['años', 'sobrevive'], axis=1)
y = titanic['años']

# División de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Configuración de la cuadrícula de hiperparámetros para Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],          # Número de árboles en el bosque
    'max_depth': [5, 10, 15, None],          # Profundidad máxima de los árboles
    'min_samples_split': [2, 5, 10],         # Mínimo de muestras para dividir un nodo
    'min_samples_leaf': [1, 2, 4]            # Mínimo de muestras en cada hoja
}

# Inicializar el modelo Random Forest para regresión
rf_model = RandomForestRegressor(random_state=42)

# Usar GridSearchCV para optimizar los hiperparámetros
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train_scaled, y_train)

# Mejor modelo encontrado
best_rf_model = grid_search.best_estimator_

# Resultados del modelo optimizado
st.subheader("Resultados del Random Forest Regressor Optimizado")
y_train_pred_opt = best_rf_model.predict(X_train_scaled)
y_test_pred_opt = best_rf_model.predict(X_test_scaled)
st.write(f"Error cuadrático medio (MSE) en entrenamiento: {mean_squared_error(y_train, y_train_pred_opt):.2f}")
st.write(f"Error cuadrático medio (MSE) en prueba: {mean_squared_error(y_test, y_test_pred_opt):.2f}")
st.write(f"Coeficiente de determinación (R²) en entrenamiento: {r2_score(y_train, y_train_pred_opt):.2f}")
st.write(f"Coeficiente de determinación (R²) en prueba: {r2_score(y_test, y_test_pred_opt):.2f}")

# Validación cruzada para verificar estabilidad del modelo
cv_scores = cross_val_score(best_rf_model, X, y, cv=10, scoring='r2')
st.write(f"Coeficiente de determinación promedio (R²) con validación cruzada: {np.mean(cv_scores):.2f}")

# Gráfico de comparación entre valores reales y predichos
st.subheader("Comparación de Valores Reales y Predichos (Conjunto de Prueba)")
fig, ax = plt.subplots()
ax.scatter(y_test, y_test_pred_opt, edgecolor='k', alpha=0.7)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
ax.set_xlabel("Valor Real")
ax.set_ylabel("Valor Predicho")
ax.set_title("Valores Reales vs. Predichos")
st.pyplot(fig)

# Apartado para ingresar datos y hacer predicciones
st.subheader("Hacer una Predicción de Edad")

# Crear formulario de entrada de usuario
with st.form("prediction_form"):
    clase_social = st.selectbox("Clase Social", [1, 2, 3])
    esposos_hermanos = st.slider("Esposos/Hermanos A Bordo", 0, 8, 0)
    padres_hijos = st.slider("Padres/Hijos A Bordo", 0, 8, 0)
    sexo = st.selectbox("Sexo", ["Femenino", "Masculino"])
    
    # Transformar datos de entrada en formato adecuado
    sexo_valor = 1 if sexo == "Masculino" else 0
    input_data = pd.DataFrame({
        'clase_social': [clase_social],
        'esposos_hermanos': [esposos_hermanos],
        'padres_hijos': [padres_hijos],
        'sexo_male': [sexo_valor]
    })
    
    # Escalar los datos de entrada para que coincidan con el entrenamiento
    input_data_scaled = scaler.transform(input_data)
    
    # Botón para hacer la predicción
    submit = st.form_submit_button("Predecir Edad")

    if submit:
        age_prediction = best_rf_model.predict(input_data_scaled)[0]
        st.write(f"La edad predicha es: {age_prediction:.2f} años")