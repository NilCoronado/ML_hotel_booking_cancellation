<div align="center">

# 🏨 Hotel Booking Cancellation Prediction

**Predicción de cancelaciones de reservas hoteleras mediante Machine Learning**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![status](https://img.shields.io/badge/status-en%20desarrollo-yellow)]()

*Un modelo de clasificación binaria que anticipa qué reservas tienen alto riesgo de cancelarse, permitiendo a los hoteles optimizar el overbooking y proteger sus ingresos.*

</div>

---

## 📋 Tabla de contenidos

- [Descripción del problema](#-descripción-del-problema)
- [Dataset](#-dataset)
- [Objetivo del modelo](#-objetivo-del-modelo)
- [Metodología](#-metodología)
- [Resultados](#-resultados)
- [Estructura del repositorio](#-estructura-del-repositorio)
- [Tecnologías](#-tecnologías)
- [Instalación y uso](#-instalación-y-uso)
- [Limitaciones y próximos pasos](#-limitaciones-y-próximos-pasos)
- [Autores](#-autores)

---

## 🎯 Descripción del problema

Las cancelaciones de última hora son uno de los principales quebraderos de cabeza en la gestión hotelera: dejan habitaciones vacías sin margen para revenderlas, erosionando directamente los ingresos. Este proyecto aborda una pregunta muy concreta:

> **¿Podemos anticipar, en el momento de la reserva, qué probabilidad tiene de cancelarse?**

Un modelo capaz de responder esto con fiabilidad permite decisiones de negocio accionables: overbooking controlado, políticas de depósito diferenciadas y campañas de reconfirmación dirigidas específicamente a las reservas de mayor riesgo — en lugar de tratar a todos los clientes por igual.

## 📊 Dataset

| | |
|---|---|
| **Fuente** | [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) (Kaggle, público) |
| **Volumen** | 119.390 reservas |
| **Periodo** | 2015 – 2017 |
| **Hoteles** | City Hotel y Resort Hotel |
| **Variable objetivo** | `is_canceled` (0 = no cancela · 1 = cancela) |

## 🎯 Objetivo del modelo

Desarrollar un modelo de **clasificación binaria supervisada** que prediga `is_canceled`, anticipando el riesgo de cancelación antes de la fecha de llegada del huésped.

### Preguntas que responde el proyecto

- ¿Qué variables predicen mejor una cancelación — antelación de la reserva, tarifa, peticiones especiales, procedencia del huésped?
- ¿Con qué fiabilidad puede el modelo identificar reservas de alto riesgo?
- ¿Cómo se traduce esta predicción en una ventaja operativa real para la gestión de ingresos del hotel?


## 🔬 Metodología

El pipeline sigue un flujo supervisado completo, con especial rigor en dos puntos críticos: la prevención de fuga de datos y la separación estricta entre entrenamiento y evaluación.

```
EDA dirigido → Preprocesado → Comparativa de modelos → Optimización → Evaluación final
```

**1. EDA dirigido al target.** Se detectaron y eliminaron `reservation_status` y `reservation_status_date` — dos columnas que revelaban el desenlace de la reserva y habrían inflado artificialmente cualquier métrica (*data leakage*).

**2. Preprocesado.** Limpieza de duplicados, imputación de nulos, *feature engineering* (variables derivadas como antelación agrupada o duración total de estancia), encoding y escalado — todo ajustado exclusivamente sobre el conjunto de entrenamiento.

**3. Comparativa de modelos.** Cinco algoritmos evaluados por validación cruzada: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting y KNN.

**4. Selección de métrica.** Con un desbalanceo de clases moderado (~37% de cancelaciones), se eligió **F1** como métrica principal, complementada con **Recall** como métrica secundaria por su valor operativo directo: a un hotel le interesa sobre todo no dejar escapar cancelaciones reales.

**5. Optimización.** El modelo ganador, **Random Forest**, se afinó mediante `RandomizedSearchCV`.

**6. Evaluación final.** Una única evaluación contra el conjunto de test, nunca antes tocado, para obtener una estimación honesta del rendimiento en producción.

## 📈 Resultados

Evaluación final sobre el conjunto de test — **23.878 reservas** no vistas durante el entrenamiento:

| Métrica | Valor |
|---|:---:|
| **F1** | `0.838` |
| **Recall** | `0.791` |
| **Precision** | `0.891` |
| **Accuracy** | `0.887` |
| **ROC-AUC** | `0.954` |

> El modelo detecta **el 79% de las cancelaciones reales** con un **89% de precisión**, lo que permite anticipar la gran mayoría de cancelaciones generando pocas falsas alarmas — la base ideal para decisiones de overbooking controlado y campañas de reconfirmación dirigidas a las reservas de mayor riesgo.

**Variables más predictivas:** `lead_time` (antelación de la reserva), `adr` (tarifa diaria), `total_of_special_requests` y `arrival_date_day_of_month`.

---

## 🗂 Estructura del repositorio

```
├── src/
│   ├── data_sample/    # Muestra de datos (máx. 100 MB)
│   ├── img/            # Gráficas del proyecto
│   ├── models/         # Modelos y transformadores entrenados (joblib)
│   ├── notebooks/      # Notebooks de desarrollo (EDA, preprocesado, modelado)
│   ├── utils/          # Funciones auxiliares reutilizables
├── main.ipynb          # Notebook final del pipeline completo
├── Presentacion.pdf    # Documento de soporte de la exposición
├── README.md
└── requirements.txt
```

## 🛠 Tecnologías

`Python` · `pandas` · `numpy` · `scikit-learn` · `matplotlib` · `seaborn` · `Jupyter`

## ⚙️ Instalación y uso

```bash
# Clona el repositorio
git clone https://github.com/NilCoronado/ML_hotel_booking_cancellation.git
cd ML_hotel_booking_cancellation

# Instala las dependencias
pip install -r requirements.txt

# Ejecuta el pipeline completo
jupyter notebook main.ipynb
```

## ⚠️ Limitaciones y próximos pasos

**Limitaciones.** Los datos corresponden a dos hoteles concretos entre 2015 y 2017, por lo que el modelo podría no generalizar bien a otros mercados o periodos. Se detectó además una diferencia en el balance de clases entre train (27,5% de cancelaciones) y test (37%) tras la eliminación de duplicados en el preprocesado, lo que pudo influir levemente en el aprendizaje del modelo.

**Próximos pasos.**
- Ajustar el umbral de decisión para priorizar aún más el recall.
- Explorar técnicas de reequilibrado de clases (SMOTE).
- Incorporar datos más recientes para mejorar la capacidad de generalización.

## 👥 Autores

- **Nil Coronado**
- **Lucía Vetrano**
