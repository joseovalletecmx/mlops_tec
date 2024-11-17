# Modelo Desplegado con FastAPI y Docker

## Requisitos

- Python 3.9+
- Docker

## Instrucciones

1. Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Ejecutar localmente

```bash
uvicorn main:app --reload
```

## Construir y correr el contenedor Docker

```bash
docker build -t model_api_image .
docker run -d -p 8000:8000 model_api_image
```
