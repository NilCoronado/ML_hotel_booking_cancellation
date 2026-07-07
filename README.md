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
_Pendiente de completar tras el desarrollo del modelo._

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
_Pendiente de completar tras la evaluación final del modelo._

## Autores
- Nil Coronado
- Lucía Vetrano
