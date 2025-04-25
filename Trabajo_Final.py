import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import openpyxl
from io import BytesIO
import requests

# Cargar los datos
file_path = "C:/Users/kema0/OneDrive/Documents/Maestria/4to Semestre/Seminario de profundizacion 1/Trabajo final/Students_Grading_Dataset.csv"
df = pd.read_csv(file_path)

st.subheader("Análisis del impacto de variables personales, académicas y conductuales sobre el rendimiento académico de los estudiantes de diferentes departamentos universitarios")

st.write(f"Este trabajo presenta un análisis exploratorio interactivo del rendimiento académico de estudiantes universitarios utilizando la plataforma Streamlit. A partir de un dataset con 5000 observaciones, se examinaron variables académicas, conductuales y personales, como horas de estudio, nivel de estrés, horas de sueño y puntajes de evaluación. Se implementaron técnicas de limpieza e imputación de datos, así como gráficos dinámicos y estadísticas descriptivas para facilitar la interpretación. ")


st.write(df.head())

# Obtener número de observaciones y columnas
n_filas, n_columnas = df.shape

# Mostrar en la interfaz
st.write(f"El dataset tiene **{n_filas} observaciones** y **{n_columnas} columnas**.")

# Variables numéricas
num_vars = df.select_dtypes(include=['number']).columns.tolist()

# Variables categóricas
cat_vars = df.select_dtypes(include=['object', 'category']).columns.tolist()

st.subheader("Tipos de variables")
st.write(df.dtypes)

st.subheader("Valores nulos por columna")
st.write(df.isnull().sum())

st.write(f"Como se puede observar, las variables Attendance (%), Assignments_Avg y Parent_Education_Level tiene valores faltantes, por tanto es necesario realizar una imputación.")


st.subheader("Imputación de valores faltantes en Attendance (%)")



import scipy.stats as stats


# Prueba de normalidad (Shapiro-Wilk)
attendance_no_na = df['Attendance (%)'].dropna()
stat, p_value = stats.shapiro(attendance_no_na)

# Interpretar la prueba
if p_value > 0.05:
    st.info(f"La variable 'Attendance (%)' parece seguir una distribución normal (p = {p_value:.4f}).")
else:
    st.warning(f"La variable 'Attendance (%)' no sigue una distribución normal (p = {p_value:.4f}). Se recomienda usar la mediana para imputar.")

# Mostrar número de valores nulos antes
n_nulos = df['Attendance (%)'].isnull().sum()
porc_nulos = (n_nulos / len(df)) * 100
st.write(f"Número de valores nulos en la variable 'Attendance (%)' antes de la imputación: {n_nulos} ({porc_nulos:.2f}%)")

# Imputar con la mediana
mediana_attendance = df['Attendance (%)'].median()
df['Attendance (%)'].fillna(mediana_attendance, inplace=True)

# Mostrar resultado de la imputación
st.success(f"Se usó la mediana: {mediana_attendance:.2f} en la imputación")
st.write(f"Número de valores nulos después de la imputación: {df['Attendance (%)'].isnull().sum()}")


st.subheader("Imputación de valores faltantes en Assignments_Avg")

import scipy.stats as stats


# Prueba de normalidad (Shapiro-Wilk)
assignments_no_na = df['Assignments_Avg'].dropna()
stat_assign, p_value_assign = stats.shapiro(assignments_no_na)

# Interpretar la prueba
if p_value_assign > 0.05:
    st.info(f"La variable 'Assignments_Avg' parece seguir una distribución normal (p = {p_value_assign:.4f}).")
else:
    st.warning(f"La variable 'Assignments_Avg' no sigue una distribución normal (p = {p_value_assign:.4f}). Se recomienda usar la mediana para imputar.")

# Calcular número y porcentaje de valores nulos antes
n_nulos_assign = df['Assignments_Avg'].isnull().sum()
porc_nulos_assign = (n_nulos_assign / len(df)) * 100
st.write(f"Número de valores nulos en 'Assignments_Avg' antes de la imputación: {n_nulos_assign} ({porc_nulos_assign:.2f}%)")

# Imputar con la mediana
mediana_assignment = df['Assignments_Avg'].median()
df['Assignments_Avg'].fillna(mediana_assignment, inplace=True)

# Mostrar resultado de la imputación
st.success(f"Se usó la mediana: {mediana_assignment:.2f} en la imputación")
st.write(f"Número de valores nulos después de la imputación: {df['Assignments_Avg'].isnull().sum()}")



st.subheader("Imputación de valores faltantes en Parent_educational_leve")

# Calcular número y porcentaje de valores nulos antes
n_nulos_parent = df['Parent_Education_Level'].isnull().sum()
porc_nulos_parent = (n_nulos_parent / len(df)) * 100

st.write(f"Dado que el porcentaje de valores faltantes es alto, es mejor no imputar con la moda, ya que más de un tercio de los valores serían reemplazados por un solo valor.")


st.write(f"Número de valores nulos en 'Parent_Education_Level' antes de la imputación: {n_nulos_parent} ({porc_nulos_parent:.2f}%)")

# Imputar con una categoría especial
df['Parent_Education_Level'].fillna("No reportado", inplace=True)

# Mostrar resultado de la imputación
st.success("Imputación completada. Se usó la categoría especial: 'No reportado'")
st.write(f"Número de valores nulos después de la imputación: {df['Parent_Education_Level'].isnull().sum()}")


import streamlit as st
import plotly.express as px
import pandas as pd

st.title("Visualización de variables numéricas")

# Obtener variables numéricas
num_vars = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Selector interactivo
var = st.selectbox("Selecciona una variable numérica:", num_vars, key="selector_numerico")

# Histograma
st.subheader(f"Histograma de {var}")
fig_hist = px.histogram(df, x=var, nbins=30, text_auto=True,
                        color_discrete_sequence=["#4C78A8"])
fig_hist.update_traces(marker_line_color='black', marker_line_width=1.2)
fig_hist.update_layout(bargap=0.1)
st.plotly_chart(fig_hist)

# Boxplot
st.subheader(f"Boxplot de {var}")
fig_box = px.box(df, y=var, points="all", title=f"Boxplot de {var}")
fig_box.update_traces(marker_line_color='black', marker_line_width=1.2)
st.plotly_chart(fig_box)


st.subheader("Matriz de correlación")
import seaborn as sns
import matplotlib.pyplot as plt

corr = df[num_vars].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.markdown("""
La variable de interes, total_Score vs cualquier otra variable muestra correlaciones que van de −0.03 a +0.02, lo cual indica que no hay ninguna relación lineal fuerte o moderada entre el puntaje total y las demás variables numéricas.
""")

import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Exploración por Departamento y Variable Numérica")

# Obtener listas
num_vars = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
departamentos = df['Department'].dropna().unique()

# Sidebar
st.sidebar.header("Opciones de análisis")

# Selección de departamento
departamento_seleccionado = st.sidebar.selectbox("Selecciona un departamento", sorted(departamentos))

# Selección de variable numérica
variable_numerica = st.sidebar.selectbox("Selecciona una variable numérica", num_vars)

# Filtrar el DataFrame
df_filtrado = df[df['Department'] == departamento_seleccionado]

st.write(f"Datos filtrados para el departamento: **{departamento_seleccionado}**")
st.write(f"Número de observaciones: {len(df_filtrado)}")

st.subheader(f"Resumen estadístico de {variable_numerica}")

# Calcular estadísticas
serie = df_filtrado[variable_numerica].dropna()

stats = {
    "Mínimo": f"{serie.min():.2f}",
    "Máximo": f"{serie.max():.2f}",
    "Media": f"{serie.mean():.2f}",
    "Mediana": f"{serie.median():.2f}",
    "Desviación estándar": f"{serie.std():.2f}",
    "Varianza": f"{serie.var():.2f}"
}

# Mostrar en formato tipo tablero
keys = list(stats.keys())
for i in range(0, len(keys), 3):  # Mostrar de a 3 por fila
    cols = st.columns(3)
    for j in range(3):
        if i + j < len(keys):
            key = keys[i + j]
            cols[j].metric(label=key, value=stats[key])


# Histograma con color individual por valor (conversión a string para colorear)
import plotly.express as px

# Convertir a int (opcional si ya lo es)
df_filtrado[variable_numerica] = df_filtrado[variable_numerica].astype(int)

# Agrupar y contar
conteo = df_filtrado[variable_numerica].value_counts().sort_index().reset_index()
conteo.columns = [variable_numerica, 'count']

fig_bar = px.bar(conteo, x=variable_numerica, y='count', text='count',
                 title=f"Distribución de {variable_numerica}",
                 labels={variable_numerica: variable_numerica, 'count': 'Frecuencia'})
fig_bar.update_traces(marker_line_color='black', marker_line_width=1.2)
fig_bar.update_layout(bargap=0.1)

st.plotly_chart(fig_bar)




# Boxplot
st.subheader(f"Boxplot de {variable_numerica}")
fig_box = px.box(df_filtrado, y=variable_numerica, points="all", title=f"Boxplot de {variable_numerica}")
st.plotly_chart(fig_box)


st.subheader("Gráfico de dispersión de las variables que podrian influir en el Puntaje Total")

# Opciones disponibles para el eje X
opciones_x = ['Study_Hours_per_Week', 'Stress_Level (1-10)', 'Sleep_Hours_per_Night']
x_var = st.selectbox("Selecciona la variable para el eje X", opciones_x, index=0, key="scatter_x_var")

# Crear gráfico con escala continua de color (por Total_Score)
fig = px.scatter(
    df_filtrado,
    x=x_var,
    y='Total_Score',
    color='Total_Score',  # ← usamos total score para relleno
    color_continuous_scale='Viridis',
    title=f"Relación entre {x_var} y Total_Score",
    opacity=0.7
)

# Ajuste si la variable X es categórica
if df_filtrado[x_var].dtype == 'object' or df_filtrado[x_var].dtype.name == 'category':
    fig.update_xaxes(type='category')

st.plotly_chart(fig)


corr = df_filtrado[[x_var, 'Total_Score']].corr().iloc[0,1]


if abs(corr) < 0.1:
    st.markdown(f"En el grafico de dispercion (que muestra la relacion entre las **{x_var}** y `Total_Score`) se observa que no existe una tendencia clara: los puntos están bastante dispersos, lo que sugiere que dormir más horas no garantiza un mayor rendimiento académico. Esto sugiere que otras variables, podrían estar influyendo más significativamente en el rendimiento que solo la cantidad de horas dedicadas al estudio.")
elif corr > 0.3:
    st.markdown(f"Existe una relación **positiva moderada** entre **{x_var}** y `Total_Score` (r = {corr:.2f}). Un aumento en {x_var} podría estar asociado con un mayor puntaje.")
elif corr < -0.3:
    st.markdown(f"Existe una relación **negativa moderada** entre **{x_var}** y `Total_Score` (r = {corr:.2f}). Un aumento en {x_var} podría estar relacionado con una disminución en el puntaje.")
else:
    st.markdown(f"La relación entre **{x_var}** y `Total_Score` es débil (r = {corr:.2f}). Podría existir cierta tendencia, pero no es concluyente.")



st.subheader("Conclusiones")

st.markdown("""
Aunque puede haber una percepción intuitiva de que dormir y estudiar más horas, asi como un bajo nivel de estres podría ayudar al rendimiento, en este dataset no se observa tal efecto directo.
""")

st.markdown("""
El rendimiento académico es posiblemente afectado por la interacción de múltiples factores que no se capturan completamente con variables simples como sueño, horas de estudio o nivel de estrés.
""")