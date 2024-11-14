# Importamos las librerías necesarias
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree
from imblearn.under_sampling import RandomUnderSampler

# Configuración de la página en Streamlit
st.set_page_config(page_title="Predicción de Supervivencia en el Titanic", layout="wide")
st.title("Predicción de Supervivencia en el Titanic")

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
    df.rename(columns={'sexo_masculino': 'sexo'}, inplace=True)
    return df

# Cargar y procesar los datos
titanic = load_data()
titanic = preprocess_data(titanic)

# Mostrar los primeros registros de los datos
st.subheader("Datos del Titanic")
st.write(titanic.head())

# Balanceo de datos
undersample = RandomUnderSampler(random_state=42)
X_titanic = titanic.drop('sobrevive', axis=1)
y_titanic = titanic.sobrevive
X_balanced, y_balanced = undersample.fit_resample(X_titanic, y_titanic)

# División de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42)

# Configuración del modelo
clf = DecisionTreeClassifier(random_state=42)
param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': [2, 3, 4, 5]}
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=10, return_train_score=True)
grid_search.fit(X_train, y_train)

# Modelo optimizado
best_clf = grid_search.best_estimator_

# Resultados de la búsqueda de hiperparámetros
st.subheader("Mejores hiperparámetros encontrados")
st.write(grid_search.best_params_)
st.write(f"Mejor puntuación de validación cruzada: {grid_search.best_score_:.2f}")

# Mostrar el accuracy
st.subheader("Métricas de Precisión")
y_train_pred = best_clf.predict(X_train)
y_test_pred = best_clf.predict(X_test)
st.write(f"Accuracy en entrenamiento: {accuracy_score(y_train, y_train_pred):.2f}")
st.write(f"Accuracy en prueba: {accuracy_score(y_test, y_test_pred):.2f}")

# Matriz de Confusión
st.subheader("Matriz de Confusión")
fig, ax = plt.subplots()
cm = confusion_matrix(y_test, y_test_pred, labels=best_clf.classes_)
ConfusionMatrixDisplay(cm, display_labels=best_clf.classes_).plot(ax=ax)
st.pyplot(fig)

# Importancia de las características
st.subheader("Importancia de las Características")
feature_scores = pd.DataFrame(pd.Series(best_clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)).T
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(data=feature_scores, ax=ax)
for index, value in enumerate(feature_scores.values.flatten()):
    ax.annotate(f'{value:.2f}', xy=(index, value), ha='center', va='bottom')
plt.title("Factores clave en la predicción de la supervivencia en el Titanic")
st.pyplot(fig)

# Visualización del Árbol de Decisión completo
st.subheader("Visualización del Árbol de Decisión")
fig = plt.figure(figsize=(12, 8))
tree.plot_tree(best_clf, feature_names=X_train.columns, class_names=["No Sobrevive", "Sobrevive"], filled=True)
st.pyplot(fig)

# Apartado para ingresar datos y hacer predicciones
st.subheader("Hacer una Predicción")

# Crear formulario de entrada de usuario
with st.form("prediction_form"):
    clase_social = st.selectbox("Clase Social", [1, 2, 3])
    años = st.slider("Edad", 0, 100, 25)
    esposos_hermanos = st.slider("Esposos/Hermanos A Bordo", 0, 8, 0)
    padres_hijos = st.slider("Padres/Hijos A Bordo", 0, 8, 0)
    sexo = st.selectbox("Sexo", ["Femenino", "Masculino"])
    
    # Transformar datos de entrada en formato adecuado
    # Transformar datos de entrada en formato adecuado
    sexo_valor = 1 if sexo == "Masculino" else 0
    input_data = pd.DataFrame({
        'clase_social': [clase_social],
        'años': [años],
        'esposos_hermanos': [esposos_hermanos],
        'padres_hijos': [padres_hijos],
        'sexo_male': [sexo_valor]  # Cambia 'sexo' a 'sexo_male' para que coincida
    })

    
    # Botón para hacer la predicción
    submit = st.form_submit_button("Predecir Supervivencia")

    if submit:
        prediction = best_clf.predict(input_data)
        if prediction[0] == 1:
            st.success("La persona sobreviviría.")
        else:
            st.error("La persona no sobreviviría.")
