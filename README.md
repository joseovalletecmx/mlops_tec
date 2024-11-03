# Regensburg Pediatric Appendicitis - ML Analysis Project

## Descripción del Proyecto

Este proyecto tiene como objetivo el desarrollo de un modelo de Machine Learning basado en el análisis del dataset *Regensburg Pediatric Appendicitis*. El propósito principal es utilizar datos clínicos de pacientes pediátricos para entrenar un modelo predictivo que ayude a mejorar la precisión diagnóstica de la apendicitis en niños. El análisis se realiza aplicando mejores prácticas de gobernanza de datos, reproducibilidad y pruebas unitarias e integrales, garantizando la calidad y la ética en el uso de los datos.

## Estructura del Proyecto

El repositorio incluye los siguientes componentes principales:

- `notebooks/`: Contiene los notebooks utilizados para la limpieza de datos, el preprocesamiento, el entrenamiento y la evaluación del modelo.
- `data/`: Carpeta que incluye el dataset anonimizado y cualquier dato procesado que se utilice en los experimentos.
- `models/`: Archivos que almacenan los modelos entrenados y versionados.
- `scripts/`: Scripts de Python que automatizan el preprocesamiento de datos y el entrenamiento de los modelos.
- `docs/`: Documentación sobre la implementación, políticas de privacidad y reporte final del análisis.

## Dataset Utilizado

El dataset utilizado en este proyecto es el **Regensburg Pediatric Appendicitis**, un conjunto de datos clínicos anónimos que contiene información de pacientes pediátricos con sospecha de apendicitis. Este dataset incluye variables relevantes como:

- **Edad** del paciente.
- **Síntomas clínicos** (por ejemplo, dolor abdominal, náuseas).
- **Resultados de exámenes físicos** (por ejemplo, sensibilidad en el abdomen).
- **Pruebas diagnósticas** (por ejemplo, análisis de sangre, imágenes).
- **Diagnóstico final** de apendicitis.

### Políticas de Privacidad

El dataset ha sido anonimizado para cumplir con las regulaciones de protección de datos, como el Reglamento General de Protección de Datos (GDPR). No contiene información personal identificable directa, y todos los datos sensibles han sido procesados para minimizar el riesgo de reidentificación. El uso de este dataset está restringido a fines educativos y de investigación.

### Fuente del Dataset

El dataset fue proporcionado por el **Centro Médico de la Universidad de Regensburg** para su uso en estudios de investigación clínica. Se encuentra disponible bajo licencia para investigación y análisis científicos, pero no está autorizado para usos comerciales ni para la distribución pública sin los debidos permisos.

## Requisitos del Proyecto

Para ejecutar este proyecto, es necesario contar con las siguientes bibliotecas y entornos:

- **Python 3.8+**
- **pandas**
- **scikit-learn**
- **ydata-profiling** (para análisis exploratorio de datos)
- **MLflow** (para el registro de experimentos y seguimiento de modelos)
- **ipywidgets** (para la visualización de perfiles de datos en notebooks)
- **matplotlib** y **seaborn** (para visualizaciones)
- **DVC** (opcional, para control de versiones de datos)

### Instalación de Dependencias

Puedes instalar todas las dependencias ejecutando el siguiente comando:

```bash
pip install -r requirements.txt