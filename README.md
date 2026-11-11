#  Predicción de diabetes con minería de datos

[![Abrir en Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://prediccion-diabetes-cuc.streamlit.app)

Aplicación web en Streamlit que estima si una paciente presenta diabetes a partir de indicadores clínicos y demográficos no invasivos. Integra el modelo desarrollado en el proyecto de aula del curso **Minería de Datos** (Universidad de la Costa – CUC).

**Aplicación publicada:** https://prediccion-diabetes-cuc.streamlit.app

>  **Prototipo académico.** La aplicación no es un dispositivo médico, no realiza diagnósticos y no sustituye la valoración de un profesional de la salud.

## Integrantes

-Alison Lara Correa-25786
-Roger Bolaño Truyol-25368
-Valeri Alvarez Blanco- 25158

**Curso:** Minería de Datos · **Profesor:** José Escorcia Gutiérrez · **Programa:** Ingeniería de Sistemas – Universidad de la Costa (CUC)

## Problema y objetivo

Clasificación binaria de la variable `Outcome` (1 = diabetes, 0 = no diabetes). El objetivo es encontrar el modelo con el mejor equilibrio entre **detectar a las pacientes diabéticas (recall)** y **evitar falsas alarmas (precision)**. En un uso de tamizaje, no detectar a una paciente diabética (falso negativo) es más grave que una falsa alarma, por lo que el modelo no se eligió por accuracy.

## Datos

**Pima Indians Diabetes Dataset** – National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK). 768 mujeres de ascendencia Pima de 21 años o más, 8 variables predictoras y la variable objetivo.

- Descarga: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv
- Referencia en Kaggle: https://www.kaggle.com/datasets/amineipad/diabetes-dataset

| Variable | Descripción | Unidad |
|---|---|---|
| `pregnancies` | Número de embarazos | embarazos |
| `glucose` | Glucosa en plasma 2 h después de una prueba oral de tolerancia | mg/dL |
| `bloodpressure` | Presión arterial diastólica | mm Hg |
| `skinthickness` | Pliegue cutáneo del tríceps | mm |
| `insulin` | Insulina sérica a las 2 horas | µU/mL |
| `bmi` | Índice de masa corporal | kg/m² |
| `diabetes_pedigree_function` | Puntaje de antecedentes familiares de diabetes | puntaje |
| `age` | Edad | años |

En `glucose`, `bloodpressure`, `skinthickness`, `insulin` y `bmi` los ceros son datos faltantes (no son valores fisiológicamente posibles). El pipeline los imputa con la mediana del conjunto de entrenamiento.

## Modelo final

**Random Forest optimizado** dentro de un `Pipeline` de scikit-learn:

1. `ColumnTransformer`: convierte los ceros y los campos vacíos de las variables clínicas en faltantes y los imputa con la mediana del entrenamiento.
2. `RandomForestClassifier(class_weight="balanced", max_depth=8, max_features=0.5, min_samples_leaf=10, n_estimators=300)`.

Hiperparámetros elegidos con `GridSearchCV` (validación cruzada estratificada de 5 particiones sobre el entrenamiento, métrica F1). Partición 80/20 estratificada: 614 registros de entrenamiento y 154 de prueba.

| Métrica (clase diabetes) | Prueba (154 registros) | Validación cruzada (5 particiones) |
|---|---|---|
| Accuracy | 0.753 | 0.775 ± 0.020 |
| Precision | 0.625 | 0.653 ± 0.034 |
| Recall | 0.741 | 0.766 ± 0.027 |
| F1-score | 0.678 | 0.704 ± 0.020 |
| AUC-ROC | 0.830 | 0.838 ± 0.022 |

Matriz de confusión en prueba: 76 VN, 24 FP, 14 FN, 40 VP.

**Cambio frente al segundo avance:** el modelo elegido era la regresión logística balanceada (recall 0.704, F1 0.650, 16 diabéticas no detectadas). Tras revisar el pipeline y ajustar hiperparámetros, el Random Forest optimizado obtuvo el mayor recall y F1 en validación cruzada y en prueba, y redujo el sobreajuste del Random Forest original (accuracy de entrenamiento de 1.000 a 0.834). El detalle está en las secciones 15 a 17 del notebook.

## Contenido de la aplicación

| Pestaña | Contenido |
|---|---|
| Proyecto | Presentación, objetivo, pregunta, integrantes, fuente de los datos y variables |
| Exploración de datos | Balance de clases, glucosa por diagnóstico y tasa de diabetes por IMC o edad |
| Predicción | Formulario con validación de rangos, tipos y categorías; predicción y probabilidad (`predict_proba`) |
| Desempeño del modelo | Métricas finales, validación cruzada, matriz de confusión, importancia de variables, comparación antes/después y limitaciones |
| Verificación | Compara la predicción de la app con la del notebook para los 154 registros de prueba |

**Validación de la entrada:**
- Rangos válidos por variable (por ejemplo, glucosa de 40 a 300 mg/dL, IMC de 15 a 70, edad de 21 a 100 años). Un valor fuera de rango muestra un error y no se calcula la predicción.
- Tipos: embarazos y edad deben ser números enteros.
- Categorías: el modelo solo se aplica a mujeres; si se selecciona otra categoría se rechaza la entrada.
- Datos opcionales: presión, pliegue cutáneo e insulina pueden marcarse como «No disponible» y se imputan con la mediana del entrenamiento.
- Valores válidos pero fuera del rango visto en el entrenamiento generan una advertencia.

## Estructura del repositorio

```
├── app.py                               # Aplicación de Streamlit
├── requirements.txt                     # Dependencias con versiones fijas
├── README.md
├── Proyecto_Diabetes_Avance3.ipynb      # Notebook actualizado (EDA, modelado, optimización y guardado)
├── diabetes_dataset.csv                 # Dataset original
├── pruebas_app.py                       # Pruebas automáticas de la aplicación
└── modelo/
    ├── pipeline_diabetes.joblib         # Pipeline completo entrenado (preprocesamiento + modelo)
    ├── metricas.json                    # Métricas, matriz de confusión, medianas y rangos de entrenamiento
    └── registros_prueba.csv             # Registros de prueba con la predicción del notebook
```

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

La aplicación carga `modelo/pipeline_diabetes.joblib` y no vuelve a entrenar el modelo. El archivo se generó con **scikit-learn 1.9.1**; si se vuelve a ejecutar el notebook con otra versión (por ejemplo en Google Colab), hay que actualizar esa versión en `requirements.txt`.

Para ejecutar el notebook se necesitan además `matplotlib` y `seaborn`.

## Pruebas de funcionamiento

Ejecutar `python pruebas_app.py` (usa `streamlit.testing`). Resultados:

| Prueba | Entrada | Resultado esperado (notebook) | Resultado en la app |
|---|---|---|---|
| Caso normal | 30 años, 2 embarazos, glucosa 110, IMC 27.5, DPF 0.35, presión 70, pliegue 25, insulina 100 | Clase 0, probabilidad 0.267 | Clase 0 – No diabetes, 26.7 %  |
| Datos no disponibles | 45 años, 5 embarazos, glucosa 165, IMC 36, DPF 0.60, presión 76, pliegue e insulina «No disponible» | Clase 1, probabilidad 0.918 | Clase 1, 91.8 %, aviso de imputación  |
| Entrada inválida | Edad 15, glucosa 0, insulina 0 | Sin predicción | Error con los 3 campos y su rango válido  |
| Categoría inválida | Sexo masculino | Sin predicción | Error: el modelo solo aplica a mujeres  |
| Fuera del rango de entrenamiento | Glucosa 250 mg/dL | Predicción con advertencia | 88.2 % y advertencia  |
| Registro de prueba, fila 408 (real: diabetes) | 8 embarazos, glucosa 197, presión 74, IMC 25.9, DPF 1.191, 39 años | Clase 1, probabilidad 0.725 | Clase 1, 72.5 %  |
| Registro de prueba, fila 44 (real: no diabetes) | 7 embarazos, glucosa 159, presión 64, IMC 27.4, DPF 0.294, 40 años | Clase 1, probabilidad 0.732 (falsa alarma) | Clase 1, 73.2 %  |
| Verificación completa | 154 registros de prueba | Mismas predicciones y probabilidades | 154 de 154 coinciden  |

En las filas 408 y 44 el pliegue cutáneo y la insulina valen 0 en el dataset (faltantes), por lo que en el formulario se marcan como «No disponible».

## Despliegue en Streamlit Community Cloud

1. Repositorio público en GitHub con los archivos de esta carpeta.
2. En https://share.streamlit.io: **Create app** → **Deploy a public app from GitHub**.
3. Repositorio: este repositorio · Rama: `main` · Archivo principal: `app.py`.
4. En **Advanced settings**, elegir **Python 3.12**.
5. **Deploy**.

## Limitaciones

- No detecta 14 de 54 pacientes diabéticas del conjunto de prueba (≈ 26 %); un resultado negativo no descarta la enfermedad.
- Genera falsas alarmas: 24 de 100 pacientes sin diabetes salen como positivas (precision de 0.625).
- El conjunto de prueba es pequeño (154 registros); las métricas pueden variar con otra partición.
- El 48.7 % de los valores de insulina y el 29.6 % del pliegue cutáneo eran faltantes y se imputaron con la mediana.
- Solo incluye mujeres de ascendencia Pima de 21 años o más: los resultados no se pueden generalizar a otras poblaciones.
- Las variables muestran asociación, no causalidad.

## Uso de inteligencia artificial

| Elemento | Descripción |
|---|---|
| **Herramienta** | Claude (Anthropic). |
| **Propósito** | Revisar el pipeline del segundo avance, proponer la optimización de hiperparámetros, generar el código para guardar el pipeline con `joblib` y construir `app.py`, `requirements.txt`, `pruebas_app.py` y este README. |
| **Prompt principal** | *"Con base en el notebook del segundo avance (Proyecto_Diabetes_Avance2.ipynb), el dataset y el enunciado de la Actividad 1 del tercer corte, revisa el rendimiento del modelo, optimízalo si es necesario registrando el antes y el después, guarda el pipeline completo con joblib y construye la aplicación de Streamlit con app.py, requirements.txt y README.md para publicarla en GitHub y Streamlit Community Cloud."* |
| **Cómo se verificó** | Se re‑ejecutó el notebook completo y se confirmó que las métricas del segundo avance se reproducen; se comprobó que el nuevo preprocesamiento da exactamente las mismas predicciones que el anterior; se verificó que el `.joblib` cargado reproduce las 154 predicciones de prueba; se ejecutaron las 8 pruebas de `pruebas_app.py`; y se instaló `requirements.txt` en un entorno limpio para confirmar que la aplicación funciona solo con esas dependencias. |
| **Problemas encontrados** | (1) La conversión de ceros a faltantes estaba fuera del pipeline, así que la aplicación habría tenido que repetirla a mano. (2) Los límites de los campos numéricos de Streamlit no explican por qué se rechaza un valor. (3) Un `.joblib` creado con una versión de scikit-learn puede fallar con otra. (4) Las tablas de la aplicación mostraban decimales incompletos y columnas cortadas. |
| **Ajustes realizados** | Se movió todo el preprocesamiento al pipeline y se verificó la equivalencia; se agregó una función de validación con mensajes de error y advertencias; se fijaron las versiones en `requirements.txt`; se formatearon las tablas; y el modelo final se eligió por F1 y recall en validación cruzada, no por accuracy. |

Todos los integrantes revisaron el código y son responsables de su contenido.
