"""
Predicción de diabetes con minería de datos – Aplicación en Streamlit.

Carga el pipeline entrenado en el notebook (preprocesamiento + Random Forest optimizado)
desde modelo/pipeline_diabetes.joblib. La aplicación no vuelve a entrenar el modelo.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn
import streamlit as st

# ---------------------------------------------------------------------------
# Rutas y constantes
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent
RUTA_MODELO = BASE / "modelo" / "pipeline_diabetes.joblib"
RUTA_METRICAS = BASE / "modelo" / "metricas.json"
RUTA_PRUEBA = BASE / "modelo" / "registros_prueba.csv"
RUTA_DATOS = BASE / "diabetes_dataset.csv"

INTEGRANTES = ["Alison Lara Correa-25786", "Roger Bolaño Truyol-25368 ", "Valeri Alvarez Blanco- 25158 "]

AZUL = "#2a78d6"      # No diabetes
NARANJA = "#eb6834"   # Diabetes
COLORES_CLASE = {"No diabetes": AZUL, "Diabetes": NARANJA}
ETIQUETA_CLASE = {0: "No diabetes", 1: "Diabetes"}

# Variables que usa el modelo, en el mismo orden del entrenamiento.
# min/max: rango válido que acepta la aplicación (valores fisiológicamente posibles).
VARIABLES = {
    "pregnancies": {
        "etiqueta": "Número de embarazos", "unidad": "embarazos", "min": 0, "max": 20,
        "entero": True, "opcional": False, "defecto": 2,
        "ayuda": "Total de embarazos que ha tenido la paciente. 0 es un valor válido.",
    },
    "glucose": {
        "etiqueta": "Glucosa en plasma", "unidad": "mg/dL", "min": 40, "max": 300,
        "entero": False, "opcional": False, "defecto": 110.0,
        "ayuda": "Glucosa medida 2 horas después de una prueba oral de tolerancia a la glucosa.",
    },
    "bloodpressure": {
        "etiqueta": "Presión arterial diastólica", "unidad": "mm Hg", "min": 30, "max": 140,
        "entero": False, "opcional": True, "defecto": 70.0,
        "ayuda": "Valor inferior (diastólico) de la presión arterial.",
    },
    "skinthickness": {
        "etiqueta": "Pliegue cutáneo del tríceps", "unidad": "mm", "min": 5, "max": 100,
        "entero": False, "opcional": True, "defecto": 25.0,
        "ayuda": "Espesor del pliegue de piel en la parte posterior del brazo.",
    },
    "insulin": {
        "etiqueta": "Insulina sérica a las 2 horas", "unidad": "µU/mL", "min": 10, "max": 900,
        "entero": False, "opcional": True, "defecto": 100.0,
        "ayuda": "Insulina en sangre 2 horas después de la prueba de tolerancia a la glucosa.",
    },
    "bmi": {
        "etiqueta": "Índice de masa corporal (IMC)", "unidad": "kg/m²", "min": 15, "max": 70,
        "entero": False, "opcional": False, "defecto": 27.5,
        "ayuda": "IMC = peso (kg) / talla (m)². Ejemplo: 70 kg y 1,60 m → 27,3.",
    },
    "diabetes_pedigree_function": {
        "etiqueta": "Antecedentes familiares (DPF)", "unidad": "puntaje", "min": 0.05, "max": 2.5,
        "entero": False, "opcional": False, "defecto": 0.35,
        "ayuda": "Diabetes Pedigree Function: puntaje que resume los antecedentes de diabetes en la "
                 "familia. Valores típicos entre 0,1 y 1,0; mediana del dataset ≈ 0,37.",
    },
    "age": {
        "etiqueta": "Edad", "unidad": "años", "min": 21, "max": 100,
        "entero": True, "opcional": False, "defecto": 30,
        "ayuda": "El modelo se entrenó con mujeres de 21 años o más.",
    },
}
SEXOS = ["Femenino", "Masculino"]

st.set_page_config(page_title="Predicción de diabetes", page_icon=":material/monitor_heart:", layout="wide")


# ---------------------------------------------------------------------------
# Carga de recursos (se ejecuta una sola vez gracias a la caché)
# ---------------------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    return joblib.load(RUTA_MODELO)


@st.cache_data
def cargar_metricas():
    with open(RUTA_METRICAS, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def cargar_datos():
    df = pd.read_csv(RUTA_DATOS)
    df["Diagnóstico"] = df["Outcome"].map(ETIQUETA_CLASE)
    return df


@st.cache_data
def cargar_registros_prueba():
    return pd.read_csv(RUTA_PRUEBA)


# ---------------------------------------------------------------------------
# Validación de la entrada
# ---------------------------------------------------------------------------
def es_faltante(valor):
    return valor is None or (isinstance(valor, float) and np.isnan(valor))


def validar_entrada(valores, sexo, rangos_entrenamiento):
    """Devuelve (errores, advertencias). Si hay errores no se hace la predicción."""
    errores, advertencias = [], []

    if sexo not in SEXOS:
        errores.append("Sexo: seleccione una de las categorías disponibles.")
    elif sexo != "Femenino":
        errores.append("Sexo: el modelo se entrenó solo con mujeres, por lo que no es aplicable a hombres.")

    for var, cfg in VARIABLES.items():
        valor = valores.get(var)
        nombre = f"{cfg['etiqueta']} ({cfg['unidad']})"

        if es_faltante(valor):
            if not cfg["opcional"]:
                errores.append(f"{nombre}: es un dato obligatorio.")
            continue
        if isinstance(valor, bool) or not isinstance(valor, (int, float, np.integer, np.floating)):
            errores.append(f"{nombre}: debe ser un valor numérico.")
            continue
        if cfg["entero"] and float(valor) != int(valor):
            errores.append(f"{nombre}: debe ser un número entero.")
            continue
        if not cfg["min"] <= valor <= cfg["max"]:
            mensaje = f"{nombre}: {valor:g} está fuera del rango válido ({cfg['min']:g} a {cfg['max']:g})."
            if valor == 0 and cfg["opcional"]:
                mensaje += " Si no tiene este dato, marque «No disponible»."
            errores.append(mensaje)
            continue

        rango = rangos_entrenamiento[var]
        if not rango["min"] <= valor <= rango["max"]:
            advertencias.append(
                f"{nombre}: {valor:g} está fuera del rango observado en el entrenamiento "
                f"({rango['min']:g} a {rango['max']:g}). La predicción es menos confiable."
            )

    embarazos, edad = valores.get("pregnancies"), valores.get("age")
    if not errores and embarazos is not None and edad is not None and embarazos > edad - 12:
        advertencias.append("El número de embarazos es poco usual para la edad indicada; revise los datos.")

    return errores, advertencias


def a_dataframe(valores):
    """Convierte el diccionario del formulario en la fila que espera el pipeline."""
    fila = {var: (np.nan if es_faltante(valores.get(var)) else float(valores[var])) for var in VARIABLES}
    return pd.DataFrame([fila], columns=list(VARIABLES))


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------
def estilo(fig, alto=380):
    fig.update_layout(height=alto, margin=dict(l=10, r=10, t=50, b=10),
                      legend_title_text="", hoverlabel=dict(font_size=13))
    return fig


def grafica_balance(df):
    conteo = df["Diagnóstico"].value_counts().reindex(["No diabetes", "Diabetes"]).reset_index()
    conteo.columns = ["Diagnóstico", "Pacientes"]
    conteo["Porcentaje"] = conteo["Pacientes"] / conteo["Pacientes"].sum() * 100
    fig = px.bar(conteo, x="Diagnóstico", y="Pacientes", color="Diagnóstico",
                 color_discrete_map=COLORES_CLASE, text=conteo["Porcentaje"].map("{:.1f} %".format),
                 title="Distribución de la variable objetivo (Outcome)")
    fig.update_traces(textposition="outside", hovertemplate="%{x}: %{y} pacientes<extra></extra>")
    fig.update_layout(showlegend=False, yaxis_title="Número de pacientes", xaxis_title="")
    return estilo(fig)


def grafica_glucosa(df):
    datos = df[df["Glucose"] > 0]
    fig = px.histogram(datos, x="Glucose", color="Diagnóstico", color_discrete_map=COLORES_CLASE,
                       category_orders={"Diagnóstico": ["No diabetes", "Diabetes"]},
                       barmode="overlay", opacity=0.65, nbins=30,
                       title="Glucosa en plasma según el diagnóstico")
    fig.update_layout(xaxis_title="Glucosa (mg/dL)", yaxis_title="Número de pacientes")
    return estilo(fig)


def grafica_tasas(df, agrupar_por):
    datos = df.copy()
    if agrupar_por == "Categoría de IMC":
        datos = datos[datos["BMI"] > 0]
        datos["Grupo"] = pd.cut(datos["BMI"], bins=[0, 25, 30, 35, 100], right=False,
                                labels=["Normal (<25)", "Sobrepeso (25-30)", "Obesidad I (30-35)", "Obesidad II+ (≥35)"])
    else:
        datos["Grupo"] = pd.cut(datos["Age"], bins=[20, 30, 40, 50, 90], labels=["21-30", "31-40", "41-50", "51+"])
    tabla = datos.groupby("Grupo", observed=True)["Outcome"].agg(["mean", "size"]).reset_index()
    tabla["Tasa"] = tabla["mean"] * 100
    fig = px.bar(tabla, x="Grupo", y="Tasa", text=tabla["Tasa"].map("{:.1f} %".format),
                 custom_data=["size"], title=f"Porcentaje de pacientes con diabetes por {agrupar_por.lower()}")
    fig.update_traces(marker_color=NARANJA, textposition="outside",
                      hovertemplate="%{x}: %{y:.1f} % con diabetes<br>%{customdata[0]} pacientes<extra></extra>")
    fig.update_layout(yaxis_title="% con diabetes", xaxis_title="", yaxis_range=[0, 70])
    return estilo(fig)


def grafica_matriz(matriz):
    z = [[matriz["VN"], matriz["FP"]], [matriz["FN"], matriz["VP"]]]
    texto = [[f"VN<br>{matriz['VN']}", f"FP (falsa alarma)<br>{matriz['FP']}"],
             [f"FN (no detectada)<br>{matriz['FN']}", f"VP<br>{matriz['VP']}"]]
    fig = go.Figure(go.Heatmap(
        z=z, x=["Predicción: No diabetes", "Predicción: Diabetes"], y=["Real: No diabetes", "Real: Diabetes"],
        text=texto, texttemplate="%{text}", textfont=dict(size=15),
        colorscale=[[0, "#cde2fb"], [1, "#1c5cab"]], showscale=False,
        hovertemplate="%{y} · %{x}<br>%{z} pacientes<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="Matriz de confusión – conjunto de prueba (154 pacientes)")
    return estilo(fig, alto=400)


def grafica_importancias(importancias):
    serie = pd.Series(importancias).rename(index={k: VARIABLES[k]["etiqueta"] for k in importancias}).sort_values()
    fig = px.bar(x=serie.values, y=serie.index, orientation="h", title="Importancia de las variables en el modelo")
    fig.update_traces(marker_color=AZUL, hovertemplate="%{y}: %{x:.3f}<extra></extra>")
    fig.update_layout(xaxis_title="Importancia (reducción media de impureza)", yaxis_title="")
    return estilo(fig, alto=400)


# ---------------------------------------------------------------------------
# Inicio de la aplicación
# ---------------------------------------------------------------------------
try:
    modelo = cargar_modelo()
    metricas = cargar_metricas()
    datos = cargar_datos()
    registros_prueba = cargar_registros_prueba()
except FileNotFoundError as error:
    st.error(f"No se encontró un archivo necesario: {error.filename}")
    st.stop()

st.title("Predicción de diabetes con minería de datos")
st.caption("Proyecto de aula · Minería de Datos · Universidad de la Costa (CUC)")
st.warning(
    "**Prototipo académico.** Esta aplicación no es un dispositivo médico, no realiza diagnósticos "
    "y no sustituye la valoración de un profesional de la salud.",
    icon=":material/warning:",
)
if sklearn.__version__ != metricas["version_sklearn"]:
    st.info(f"El modelo se guardó con scikit-learn {metricas['version_sklearn']} y el servidor usa "
            f"{sklearn.__version__}. Revise que `requirements.txt` fije la misma versión.")

with st.sidebar:
    st.header("Resumen")
    st.markdown(
        f"**Modelo:** {metricas['modelo']}  \n"
        f"**Recall (diabetes):** {metricas['prueba']['recall']:.1%}  \n"
        f"**Accuracy:** {metricas['prueba']['accuracy']:.1%}  \n"
        f"**Datos:** Pima Indians Diabetes (768 pacientes)"
    )
    st.markdown("**Integrantes:**  \n" + "  \n".join(INTEGRANTES))
    st.caption("Prototipo académico. No sustituye una decisión médica.")

tab_proyecto, tab_eda, tab_prediccion, tab_modelo, tab_verificacion = st.tabs(
     ["Proyecto", "Exploración de datos", "Predicción", "Desempeño del modelo", "Verificación"]
)

# --- Proyecto ---------------------------------------------------------------
with tab_proyecto:
    col_a, col_b = st.columns([3, 2], gap="large")
    with col_a:
        st.subheader("Presentación")
        st.markdown(
            "La diabetes tipo 2 puede pasar desapercibida durante años. Este proyecto aplica técnicas de "
            "minería de datos a indicadores clínicos y demográficos no invasivos para estimar si una paciente "
            "presenta diabetes y apoyar la priorización de exámenes confirmatorios."
        )
        st.subheader("Objetivo")
        st.markdown(
            "Construir y desplegar un modelo de **clasificación binaria** que prediga la variable `Outcome` "
            "(1 = diabetes, 0 = no diabetes) buscando el mejor equilibrio entre **detectar a las pacientes "
            "diabéticas (recall)** y **evitar falsas alarmas (precision)**."
        )
        st.subheader("Pregunta de investigación")
        st.markdown(
            "¿Es posible predecir si una paciente presenta diabetes a partir de indicadores clínicos y "
            "demográficos no invasivos, y qué modelo ofrece el mejor equilibrio entre recall y precision?"
        )
    with col_b:
        st.subheader("Integrantes")
        st.markdown("\n".join(f"- {nombre}" for nombre in INTEGRANTES))
        st.markdown("**Curso:** Minería de Datos  \n**Profesor:** José Escorcia Gutiérrez  \n"
                    "**Programa:** Ingeniería de Sistemas – Universidad de la Costa (CUC)")
        st.subheader("Fuente de los datos")
        st.markdown(
            "**Pima Indians Diabetes Dataset** – National Institute of Diabetes and Digestive and Kidney "
            "Diseases (NIDDK). 768 mujeres de ascendencia Pima de 21 años o más, 8 variables predictoras y "
            "la variable objetivo `Outcome`.  \n"
            "[Descarga (GitHub)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv)"
            " · [Referencia en Kaggle](https://www.kaggle.com/datasets/amineipad/diabetes-dataset)"
        )

    st.subheader("Variables del modelo")
    st.dataframe(
        pd.DataFrame([{"Variable": var, "Descripción": cfg["etiqueta"], "Unidad": cfg["unidad"],
                       "Rango aceptado en la app": f"{cfg['min']:g} – {cfg['max']:g}",
                       "Puede faltar": "Sí" if cfg["opcional"] else "No"} for var, cfg in VARIABLES.items()]),
        hide_index=True,
    )

# --- Exploración de datos ---------------------------------------------------
with tab_eda:
    st.markdown("Visualizaciones del análisis exploratorio realizado sobre los 768 registros del dataset.")
    col_1, col_2 = st.columns(2, gap="large")
    with col_1:
        st.plotly_chart(grafica_balance(datos))
        st.caption("Hay un desbalance moderado: 65 % de las pacientes no tiene diabetes. Un modelo que siempre "
                   "respondiera «No diabetes» tendría 65 % de accuracy sin detectar a nadie, por eso el modelo "
                   "se eligió por recall y F1.")
    with col_2:
        st.plotly_chart(grafica_glucosa(datos))
        st.caption("Las pacientes con diabetes se concentran en valores de glucosa más altos (mediana 140 frente "
                   "a 107 mg/dL). La glucosa es la variable más influyente del modelo. Se excluyen 5 registros "
                   "con glucosa igual a 0 (dato faltante).")
    agrupar_por = st.radio("Agrupar la tasa de diabetes por:", ["Categoría de IMC", "Grupo de edad"], horizontal=True)
    st.plotly_chart(grafica_tasas(datos, agrupar_por))
    st.caption("La proporción de diabetes crece con el IMC (de 6,6 % con IMC normal a 47,6 % con obesidad grado II "
               "o más) y con la edad hasta los 50 años. Son asociaciones, no relaciones causales.")

# --- Predicción --------------------------------------------------------------
with tab_prediccion:
    st.subheader("Datos de la paciente")
    st.markdown("Ingrese los valores y presione **Calcular predicción**. Los campos marcados como "
                "«No disponible» se completan con la mediana del conjunto de entrenamiento.")

    with st.form("formulario_prediccion"):
        col_izq, col_der = st.columns(2, gap="large")
        valores = {}
        with col_izq:
            sexo = st.selectbox("Sexo", SEXOS, help="El modelo solo es aplicable a mujeres.")
            for var in ["age", "pregnancies", "glucose", "bmi"]:
                cfg = VARIABLES[var]
                valores[var] = st.number_input(
                    f"{cfg['etiqueta']} ({cfg['unidad']})", min_value=0 if cfg["entero"] else 0.0,
                    max_value=150 if cfg["entero"] else 1000.0, value=cfg["defecto"],
                    step=1 if cfg["entero"] else 1.0, help=cfg["ayuda"], key=f"campo_{var}")
        with col_der:
            cfg = VARIABLES["diabetes_pedigree_function"]
            valores["diabetes_pedigree_function"] = st.number_input(
                f"{cfg['etiqueta']} ({cfg['unidad']})", min_value=0.0, max_value=5.0, value=cfg["defecto"],
                step=0.01, format="%.3f", help=cfg["ayuda"], key="campo_diabetes_pedigree_function")
            no_disponible = {}
            for var in ["bloodpressure", "skinthickness", "insulin"]:
                cfg = VARIABLES[var]
                valores[var] = st.number_input(
                    f"{cfg['etiqueta']} ({cfg['unidad']})", min_value=0.0, max_value=1000.0,
                    value=cfg["defecto"], step=1.0, help=cfg["ayuda"], key=f"campo_{var}")
                no_disponible[var] = st.checkbox("No disponible", key=f"nd_{var}")
        enviado = st.form_submit_button("Calcular predicción", type="primary")

    if enviado:
        for var, marcado in no_disponible.items():
            if marcado:
                valores[var] = None

        errores, advertencias = validar_entrada(valores, sexo, metricas["rangos_entrenamiento"])
        if errores:
            st.error("**No se puede calcular la predicción. Corrija lo siguiente:**\n\n" +
                     "\n".join(f"- {e}" for e in errores))
        else:
            for aviso in advertencias:
                st.warning(aviso)

            entrada = a_dataframe(valores)
            probabilidad = float(modelo.predict_proba(entrada)[0, 1])
            clase = int(modelo.predict(entrada)[0])

            st.subheader("Resultado")
            col_r1, col_r2 = st.columns([1, 2], gap="large")
            with col_r1:
                st.metric("Probabilidad estimada de diabetes", f"{probabilidad:.1%}", border=True)
                st.progress(probabilidad, text=f"Umbral de decisión: {metricas['umbral']:.0%}")
            with col_r2:
                if clase == 1:
                    st.error(f"**Clase predicha: 1 – Diabetes.** La probabilidad supera el umbral de "
                             f"{metricas['umbral']:.0%}. En un uso de tamizaje, este resultado indicaría "
                             "priorizar una prueba confirmatoria con un profesional de la salud.", icon=":material/priority_high:")
                else:
                    st.success(f"**Clase predicha: 0 – No diabetes.** La probabilidad no supera el umbral de "
                               f"{metricas['umbral']:.0%}. Un resultado negativo no descarta la enfermedad: el "
                               f"modelo no detecta cerca de {1 - metricas['prueba']['recall']:.0%} de los casos "
                               "de diabetes.", icon=":material/check_circle:")

            imputadas = [var for var in VARIABLES if es_faltante(valores.get(var))]
            if imputadas:
                st.info("Datos no disponibles completados con la mediana del entrenamiento: " + ", ".join(
                    f"{VARIABLES[v]['etiqueta']} = {metricas['medianas_imputacion'][v]:g} {VARIABLES[v]['unidad']}"
                    for v in imputadas))

            with st.expander("Ver los datos enviados al modelo"):
                st.dataframe(entrada.rename(columns={v: VARIABLES[v]["etiqueta"] for v in VARIABLES}),
                             hide_index=True)
            st.caption("Prototipo académico: el resultado es una estimación estadística y no sustituye una "
                       "decisión profesional.")

# --- Desempeño del modelo ---------------------------------------------------
with tab_modelo:
    prueba, cv = metricas["prueba"], metricas["validacion_cruzada"]
    hp = metricas["hiperparametros"]
    st.subheader(f"Modelo final: {metricas['modelo']}")
    st.markdown(
        f"Pipeline: imputación de ceros y datos faltantes con la mediana del entrenamiento → Random Forest con "
        f"`class_weight={hp['class_weight']}`, `max_depth={hp['max_depth']}`, `max_features={hp['max_features']}`, "
        f"`min_samples_leaf={hp['min_samples_leaf']}` y `n_estimators={hp['n_estimators']}`. "
        f"Entrenado con {metricas['n_entrenamiento']} registros y evaluado con {metricas['n_prueba']} registros "
        "nunca vistos (partición 80/20 estratificada)."
    )

    st.markdown("**Métricas finales en el conjunto de prueba** (clase positiva = diabetes)")
    columnas = st.columns(5)
    nombres = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1-score",
               "roc_auc": "AUC-ROC"}
    for columna, (clave, nombre) in zip(columnas, nombres.items()):
        columna.metric(nombre, f"{prueba[clave]:.3f}", border=True,
                       help=f"Validación cruzada (5 particiones): {cv[clave]['media']:.3f} ± {cv[clave]['std']:.3f}")
    st.caption("Pase el cursor sobre el ícono de ayuda de cada métrica para ver la validación cruzada.")

    col_m1, col_m2 = st.columns(2, gap="large")
    with col_m1:
        st.plotly_chart(grafica_matriz(prueba["matriz_confusion"]))
        mc = prueba["matriz_confusion"]
        st.caption(f"De {mc['FN'] + mc['VP']} pacientes con diabetes, el modelo detecta {mc['VP']} y no detecta "
                   f"{mc['FN']}. De {mc['VN'] + mc['FP']} sin diabetes, genera {mc['FP']} falsas alarmas.")
    with col_m2:
        st.plotly_chart(grafica_importancias(metricas["importancias"]))
        st.caption("La glucosa, el IMC y la edad son las variables con mayor peso, en línea con el EDA. "
                   "La importancia indica asociación dentro del modelo, no causalidad.")

    st.markdown("**Resultados por clase**")
    por_clase = pd.DataFrame(prueba["por_clase"]).T.rename(columns={
        "precision": "Precision", "recall": "Recall", "f1-score": "F1-score", "support": "Pacientes"})
    st.dataframe(por_clase.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1-score": "{:.3f}",
                                         "Pacientes": "{:.0f}"}))

    st.markdown("**Comparación antes y después de la optimización** (mismo conjunto de prueba)")
    comparacion = pd.DataFrame(metricas["comparacion"]).rename(
        columns={"Accuracy entrenamiento": "Acc. entren.", "Accuracy prueba": "Acc. prueba",
                 "Precision (1)": "Precision", "Recall (1)": "Recall", "F1-score (1)": "F1"})
    comparacion["Etapa"] = comparacion["Etapa"].str.extract(r"\((.*)\)", expand=False)
    columnas_decimales = [c for c in comparacion.columns if c not in ["Etapa", "Modelo", "FN", "FP"]]
    st.dataframe(comparacion.style.format({c: "{:.3f}" for c in columnas_decimales}), hide_index=True)
    st.caption("En el segundo avance se eligió la regresión logística balanceada. En el tercer corte se ajustaron "
               "los hiperparámetros con GridSearchCV (validación cruzada, métrica F1) y el Random Forest "
               "optimizado obtuvo el mayor recall y F1 con menos sobreajuste.")

    st.subheader("Limitaciones del modelo")
    st.markdown(
        f"- No detecta {mc['FN']} de {mc['FN'] + mc['VP']} pacientes diabéticas del conjunto de prueba "
        f"(≈ {mc['FN'] / (mc['FN'] + mc['VP']):.0%}); un resultado negativo no descarta la enfermedad.\n"
        f"- Genera falsas alarmas: {mc['FP']} de {mc['VN'] + mc['FP']} pacientes sin diabetes salen como positivas "
        f"(precision de {prueba['precision']:.3f} para la clase diabética).\n"
        "- El conjunto de prueba es pequeño (154 registros), por lo que las métricas pueden variar con otra partición.\n"
        "- Casi la mitad de los valores de insulina (48,7 %) y el 29,6 % del pliegue cutáneo eran faltantes y "
        "se imputaron con la mediana; su aporte real es limitado.\n"
        "- Solo incluye mujeres de ascendencia Pima de 21 años o más: los resultados no se pueden generalizar "
        "a hombres ni a otras poblaciones.\n"
        "- Las variables muestran asociación, no causalidad. El modelo no reemplaza pruebas clínicas como la "
        "hemoglobina glicosilada (HbA1c) ni la valoración médica."
    )

# --- Verificación -----------------------------------------------------------
with tab_verificacion:
    st.subheader("Verificación con registros del conjunto de prueba")
    st.markdown("Cada registro de prueba se evalúa aquí con el pipeline cargado y se compara con la predicción "
                "obtenida en el notebook para el mismo registro.")

    columnas_modelo = list(VARIABLES)
    pred_app = modelo.predict(registros_prueba[columnas_modelo])
    prob_app = modelo.predict_proba(registros_prueba[columnas_modelo])[:, 1]
    coincide = (pred_app == registros_prueba["prediccion_notebook"].to_numpy()) & \
               np.isclose(prob_app, registros_prueba["probabilidad_notebook"].to_numpy(), atol=1e-6)
    st.metric("Registros cuya predicción coincide con el notebook",
              f"{int(coincide.sum())} de {len(registros_prueba)}", border=True)

    opciones = registros_prueba["fila_dataset"].tolist()
    fila = st.selectbox(
        "Registro del dataset (número de fila, empezando en 0)", opciones,
        index=opciones.index(408) if 408 in opciones else 0,
        format_func=lambda f: f"Fila {f} · valor real: "
                              f"{ETIQUETA_CLASE[int(registros_prueba.loc[registros_prueba['fila_dataset'] == f, 'outcome'].iloc[0])]}",
    )
    i = registros_prueba.index[registros_prueba["fila_dataset"] == fila][0]
    registro = registros_prueba.loc[i]
    st.dataframe(registros_prueba.loc[[i], columnas_modelo].rename(
        columns={v: VARIABLES[v]["etiqueta"] for v in VARIABLES}), hide_index=True)
    st.caption("Los ceros en presión, pliegue, insulina, glucosa o IMC son datos faltantes del dataset original; "
               "el pipeline los imputa con la mediana del entrenamiento.")

    col_v1, col_v2, col_v3 = st.columns(3)
    col_v1.metric("Valor real", ETIQUETA_CLASE[int(registro["outcome"])], border=True)
    col_v2.metric("Notebook", f"{ETIQUETA_CLASE[int(registro['prediccion_notebook'])]} · "
                              f"{registro['probabilidad_notebook']:.1%}", border=True)
    col_v3.metric("Aplicación", f"{ETIQUETA_CLASE[int(pred_app[i])]} · {prob_app[i]:.1%}", border=True)
    if coincide[i]:
        st.success("La predicción y la probabilidad de la aplicación coinciden con las del notebook.")
    else:
        st.error("La predicción de la aplicación no coincide con la del notebook. Revise las versiones de las librerías.")
