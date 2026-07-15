# ML Hotel Booking Cancellation

## Descripción del problema
Predicción de cancelación de reservas hoteleras. El objetivo es anticipar qué reservas tienen alta probabilidad de cancelarse para ayudar a los hoteles a optimizar el overbooking y la gestión de ingresos (revenue management).

## Dataset
[Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) (Kaggle, público). 119.390 reservas de dos hoteles (resort y ciudad) entre 2015 y 2017. Variable target: `is_canceled`.

## Objetivo del modelo
Desarrollar un modelo de clasificación binaria que prediga si una reserva hotelera será cancelada (`is_canceled`), permitiendo anticipar el riesgo de cancelación antes de la fecha de llegada.

### Preguntas a responder (al finalizar el proyecto)
- ¿Qué variables predicen mejor la cancelación (antelación de reserva, tipo de depósito, segmento de mercado, historial de cancelaciones previas)?
- ¿Con qué fiabilidad podemos identificar reservas de alto riesgo de cancelación?
- ¿Cómo puede esta predicción ayudar a un hotel a optimizar el overbooking y reducir la pérdida de ingresos por cancelaciones de última hora?

## Solución adoptada
Pipeline de clasificación binaria supervisada. Tras un EDA dirigido al target (que detectó y eliminó `reservation_status`/`reservation_status_date` por fuga de datos) y un preprocesado completo (limpieza de duplicados, imputación, feature engineering, encoding y escalado), se compararon varios modelos por validación cruzada (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN). **Random Forest** obtuvo el mejor F1 y se optimizó con `RandomizedSearchCV`. Se eligió F1 como métrica principal por el desbalanceo de clases (~37% de cancelaciones), reportando Recall como métrica secundaria por su valor operativo para el negocio.

## Estructura del repositorio
```
├── src/
│   ├── data_sample/    # Muestra de datos (máx. 100MB)
│   ├── img/            # Imágenes del proyecto
│   ├── models/         # Modelos entrenados (pickle/joblib)
│   ├── notebooks/      # Notebooks de desarrollo y pruebas
│   ├── utils/          # Funciones auxiliares
├── main.ipynb          # Notebook final del pipeline de ML
├── Presentacion.pdf    # Documento soporte de la exposición
├── README.md
├── requirements.txt
```

## Tecnologías utilizadas
Python, pandas, numpy, scikit-learn, matplotlib, seaborn, Jupyter.

## Instrucciones de reproducción
```bash
pip install -r requirements.txt
jupyter notebook main.ipynb
```

## Principales resultados
Evaluación final sobre el conjunto de test (23.878 reservas, no visto durante el entrenamiento):

| Métrica | Valor |
|---|---|
| F1 | 0.853 |
| Recall | 0.822 |
| Precision | 0.888 |
| Accuracy | 0.895 |
| ROC-AUC | 0.959 |

El modelo detecta el 82% de las cancelaciones reales con un 89% de precisión, lo que permite anticipar la mayoría de cancelaciones con pocas falsas alarmas y sustenta decisiones de overbooking controlado y campañas de reconfirmación dirigidas a las reservas de mayor riesgo. Variables más predictivas: `lead_time`, `deposit_type`, `market_segment` y `previous_cancellations`.

**Limitaciones:** datos de dos hoteles concretos (2015-2017), puede no generalizar a otros mercados o épocas. **Próximos pasos:** ajustar el umbral de decisión para priorizar recall, reequilibrado de clases (SMOTE), incorporar más datos recientes.

## Autores
- Nil Coronado
- Lucía Vetrano
