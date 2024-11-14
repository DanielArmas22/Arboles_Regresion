# Importamos las librerías necesarias
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from imblearn.under_sampling import RandomUnderSampler

# Configuración de la página en Streamlit
st.set_page_config(page_title="Predicción de Edad en el Titanic", layout="wide")
st.title("Predicción de Edad en el Titanic")

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
    df.rename(columns={'sexo_masculino': 'sexo_male'}, inplace=True)  # Asegura que el nombre sea correcto
    # Eliminamos filas con valores nulos en 'años'
    df.dropna(subset=['años'], inplace=True)
    return df

# Cargar y procesar los datos
titanic = load_data()
titanic = preprocess_data(titanic)

# Mostrar los primeros registros de los datos
st.subheader("Datos del Titanic")
st.write(titanic.head())

# Separar en X e y para regresión (predecir 'años')
X = titanic.drop(['años', 'sobrevive'], axis=1)  # Elimina 'sobrevive' de X
y = titanic['años']

# División de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Configuración del modelo de regresión lineal
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Resultados del modelo
st.subheader("Resultados del Modelo de Regresión Lineal")
y_train_pred = reg_model.predict(X_train)
y_test_pred = reg_model.predict(X_test)
st.write(f"Error cuadrático medio (MSE) en entrenamiento: {mean_squared_error(y_train, y_train_pred):.2f}")
st.write(f"Error cuadrático medio (MSE) en prueba: {mean_squared_error(y_test, y_test_pred):.2f}")
st.write(f"Coeficiente de determinación (R²) en entrenamiento: {r2_score(y_train, y_train_pred):.2f}")
st.write(f"Coeficiente de determinación (R²) en prueba: {r2_score(y_test, y_test_pred):.2f}")

# Coeficientes del modelo
st.subheader("Coeficientes del Modelo de Regresión")
coefficients = pd.DataFrame(reg_model.coef_, X_train.columns, columns=['Coeficiente'])
st.write(coefficients)

# Gráfico de comparación entre valores reales y predichos
st.subheader("Comparación de Valores Reales y Predichos (Conjunto de Prueba)")
fig, ax = plt.subplots()
ax.scatter(y_test, y_test_pred, edgecolor='k', alpha=0.7)
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
        'sexo_male': [sexo_valor]  # Ajusta el nombre a 'sexo_male' para que coincida con el entrenamiento
    })
    
    # Botón para hacer la predicción
    submit = st.form_submit_button("Predecir Edad")

    if submit:
        age_prediction = reg_model.predict(input_data)[0]
        st.write(f"La edad predicha es: {age_prediction:.2f} años")
