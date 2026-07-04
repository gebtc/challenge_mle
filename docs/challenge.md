# Flight Delay Prediction API — Challenge Documentation

## Índice

1. [Modelo Elegido y Justificación](#1-modelo-elegido-y-justificación)
2. [Bugs Encontrados y Correcciones](#2-bugs-encontrados-y-correcciones)
3. [Decisiones de Diseño y Arquitectura](#3-decisiones-de-diseño-y-arquitectura)
4. [Cómo Probar Cada Parte](#4-cómo-probar-cada-parte)
5. [Edge Cases Considerados](#5-edge-cases-considerados)
6. [Estructura del Proyecto](#6-estructura-del-proyecto)
7. [CI/CD](#7-cicd)
8. [Deploy en la Nube](#8-deploy-en-la-nube)

---

## 1. Modelo Elegido y Justificación

### Modelo seleccionado: **Logistic Regression** con top 10 features + class balancing

#### ¿Por qué Logistic Regression y no XGBoost?

| Criterio | Logistic Regression | XGBoost |
|----------|-------------------|---------|
| Dependencias | ✅ Ya en `requirements.txt` (scikit-learn) | ❌ Requeriría agregar `xgboost` |
| Performance | ✅ Equivalente (DS concluyó "no noticeable difference") | ✅ Equivalente |
| Velocidad de inferencia | ✅ ~0.1ms por predicción (lineal) | 🟡 ~1-5ms (ensemble de árboles) |
| Tamaño en memoria | ✅ ~50KB (coeficientes) | 🟡 ~5-50MB (árboles) |
| Interpretabilidad | ✅ Coeficientes directos por feature | 🟡 Feature importance aproximada |
| Madurez operacional | ✅ Décadas en producción | ✅ Ampliamente usado |

#### Decisión basada en el notebook del DS

El Data Scientist probó **6 modelos** en el notebook:

| # | Modelo | Features | Class Balancing | Recall "1" | F1 "0" | F1 "1" |
|---|--------|----------|----------------|-----------|--------|--------|
| 1 | XGBoost | Todas | No | - | - | - |
| 2 | Logistic Regression | Todas | No | - | - | - |
| 3 | **XGBoost + Balance** | **Top 10** | **Sí (scale_pos_weight)** | **> 0.60** | **< 0.70** | **> 0.30** |
| 4 | XGBoost | Top 10 | No | < 0.60 | - | - |
| 5 | **Logistic Regression + Balance** | **Top 10** | **Sí (class_weight)** | **> 0.60** | **< 0.70** | **> 0.30** |
| 6 | Logistic Regression | Top 10 | No | < 0.60 | - | - |

**Conclusión del DS:** "No noticeable difference in results between XGBoost and LogisticRegression."

Dado que ambos modelos producen resultados equivalentes, Logistic Regression es la opción superior por:
1. **Cero dependencias nuevas** — scikit-learn ya está en requirements.txt
2. **Menor latencia** — modelo lineal vs ensemble de árboles
3. **Menor superficie de ataque** — menos dependencias = menos vulnerabilidades
4. **Imagen Docker más pequeña** — sin compilar XGBoost

#### Estrategia de class balancing

```python
n_y0 = sum(target == 0)  # clase mayoritaria (no delay)
n_y1 = sum(target == 1)  # clase minoritaria (delay)
n_total = len(target)

class_weight = {
    1: n_y0 / n_total,  # ~0.83 (más peso a delay)
    0: n_y1 / n_total,  # ~0.17 (menos peso a no delay)
}
```

Esto replica exactamente la estrategia documentada en el notebook (sección 6.b.iii), logrando que:
- **Recall clase "0"** < 0.60 (el modelo sacrifica precisión en no-delays)
- **Recall clase "1"** > 0.60 (el modelo captura la mayoría de los delays)
- **F1 clase "1"** > 0.30

#### Top 10 Features

Seleccionadas por XGBoost feature importance (notebook sección 5):

| Feature | Tipo | Descripción |
|---------|------|-------------|
| OPERA_Latin American Wings | One-hot | Aerolínea específica |
| MES_7 | One-hot | Julio |
| MES_10 | One-hot | Octubre |
| OPERA_Grupo LATAM | One-hot | Aerolínea específica |
| MES_12 | One-hot | Diciembre |
| TIPOVUELO_I | One-hot | Vuelo Internacional |
| MES_4 | One-hot | Abril |
| MES_11 | One-hot | Noviembre |
| OPERA_Sky Airline | One-hot | Aerolínea específica |
| OPERA_Copa Air | One-hot | Aerolínea específica |

---

## 2. Bugs Encontrados y Correcciones

### Bug 1: `get_period_day()` — Límites temporales excluyentes

**Ubicación:** Notebook, celda de definición de `get_period_day()`

**Problema:** La función usa operadores de comparación estrictos (`>` y `<`) en lugar de inclusivos (`>=` y `<=`). Los tiempos exactos en los límites (05:00, 12:00, 19:00, 00:00) retornan `None` en lugar de un período válido.

```python
# ❌ Bug: if(date_time > morning_min and date_time < morning_max):
# Los vuelos exactamente a las 05:00, 12:00, 19:00, 00:00
# no caen en ningún período → retorna None
```

**Impacto:** Valores `NaN` en la columna `period_day` para vuelos con horarios exactos en los límites.

**Fix:** Este bug no afecta a `model.py` porque nuestro modelo **no usa `period_day`** como feature (no está entre las top 10). Sin embargo, está documentado en el notebook original.

### Bug 2: `get_rate_from_column()` — Tasa de delay invertida

**Ubicación:** Notebook, celda de definición de `get_rate_from_column()`

**Problema:** La función calcula `total / delays` en lugar de `delays / total`.

```python
# ❌ Bug: rates[name] = round(total / delays[name], 2)
# Esto calcula "vuelos por delay", no "tasa de delay"
# 
# ✅ Correcto: rates[name] = round(delays[name] / total, 2)
```

**Impacto:** Las gráficas de "Delay Rate by Destination/Airline/Month/..." muestran valores invertidos. Por ejemplo, si una aerolínea tiene 10 vuelos totales y 2 con delay, la tasa debería ser 20% pero el bug muestra 500%.

**Fix:** Este bug está en el notebook de exploración, no en `model.py`. No afecta el pipeline de producción, pero documentamos aquí para referencia.

### Bug 3: `Union()` con paréntesis en lugar de brackets

**Ubicación:** `challenge/model.py` línea 16 (archivo original)

**Problema:** El type hint usa `Union(...)` (paréntesis) en lugar de `Union[...]` (brackets). Esto es un error de sintaxis de Python.

```python
# ❌ Bug: Union(Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame)
# Esto lanza TypeError en tiempo de ejecución porque Union
# no es callable con paréntesis.
#
# ✅ Fix: Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]
```

**Impacto:** El archivo `model.py` original no podía ser importado sin lanzar un error de sintaxis.

**Fix:** Cambiado a `Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]`.

---

## 3. Decisiones de Diseño y Arquitectura

### Arquitectura General

```
┌───────────────────────────────────────────────────────────┐
│                      FASTAPI APP                           │
│                                                           │
│  ┌─────────────────┐   ┌──────────────┐   ┌───────────┐  │
│  │  Pydantic v1     │──▶│ Preprocessor  │──▶│    LR     │  │
│  │  Validators      │   │ (get_dummies  │   │  Model    │  │
│  │  (MES, OPERA,    │   │  + reindex    │   │  predict  │  │
│  │   TIPOVUELO)     │   │  a top 10)    │   │           │  │
│  └─────────────────┘   └──────────────┘   └───────────┘  │
│         │                                                  │
│         ▼                                                  │
│  ┌──────────────────────────────────────────────────┐      │
│  │  @app.on_event("startup")                        │      │
│  │  → Carga data.csv                                │      │
│  │  → Extrae aerolíneas conocidas                   │      │
│  │  → Entrena modelo (LogisticRegression)           │      │
│  │  → Almacena en app.state                         │      │
│  └──────────────────────────────────────────────────┘      │
└───────────────────────────────────────────────────────────┘
```

### Decisiones Clave

#### D1. Modelo entrenado al startup (Eager Loading)

El modelo se entrena una sola vez al iniciar la aplicación, no en cada request.

**Por qué:**
- Evita "thundering herd" si 100 usuarios concurrentes llegan al primer request
- Fail-fast: si el modelo no puede entrenarse, el contenedor no arranca
- El entrenamiento toma ~2 segundos para 68K filas — insignificante vs startup time

**Trade-off:** Cold start más lento (~5s), aceptable para Cloud Run.

#### D2. Aerolíneas extraídas dinámicamente de data.csv

No hay hardcoding de las 23 aerolíneas conocidas.

**Por qué:**
- Single source of truth con los datos de entrenamiento
- Si el dataset cambia, la validación se actualiza automáticamente
- Reduce riesgo de training-serving skew

#### D3. Feature alignment con `reindex(fill_value=0)`

Durante serving, los datos pueden no contener todas las columnas one-hot esperadas.

```python
features = features.reindex(columns=TOP_10_FEATURES, fill_value=0)
```

**Por qué:**
- Si un request pide una aerolínea que no está entre las top 10, su columna one-hot será 0
- Garantiza que el input de predict() siempre tenga exactamente 10 columnas
- Evita errores de shape mismatch

#### D4. Pydantic v1 con `@validator` (compatibilidad)

El challenge especifica `pydantic~=1.10.2`. Usamos sintaxis v1.

**Por qué:**
- El evaluador instalará pydantic ~1.10.2 desde requirements.txt
- v1 no tiene `@field_validator` ni `model_dump`
- En el entorno local (Python 3.14 con pydantic v2), la sintaxis v1 sigue funcionando con warnings de deprecación

#### D5. FastAPI 0.86 con `@app.on_event("startup")`

El challenge especifica `fastapi~=0.86.0`. Usamos `on_event` en lugar de `lifespan`.

**Por qué:**
- FastAPI ~0.86.0 no soporta el parámetro `lifespan` en `FastAPI()`
- `on_event` es la API correcta para esa versión
- En el entorno local (FastAPI 0.139), `on_event` funciona con warnings de deprecación

#### D6. Auto-entrenamiento en `predict()` (lazy initialization)

Si `predict()` se llama sin que `fit()` haya sido llamado antes, el modelo se entrena automáticamente.

```python
def predict(self, features):
    if self._model is None:
        # Auto-train with default dataset
        data = pd.read_csv(self._data_path)
        train_features, target = self.preprocess(data, target_column="delay")
        self.fit(train_features, target)
    return self._model.predict(features).tolist()
```

**Por qué:** El test `test_model_predict()` llama `predict()` sin `fit()` explícito.

#### D7. HTTP 400 vs 422 para errores de validación

FastAPI por defecto retorna 422 para errores de validación de pydantic. Overrideamos a 400.

```python
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(status_code=400, content={"detail": str(exc)})
```

**Por qué:** Los tests del challenge esperan status_code 400.

#### D8. Preprocesamiento sin target en serving

Cuando `preprocess()` se llama sin `target_column` (modo serving), NO intenta computar `min_diff` ni `delay`.

**Por qué:** Los datos de serving no contienen `Fecha-O` (fecha de operación), solo `Fecha-I`, `OPERA`, `TIPOVUELO`, `MES`. Intentar computar `min_diff` sin `Fecha-O` lanzaría KeyError.

---

## 4. Cómo Probar Cada Parte

### Tests Unitarios (Modelo)

```bash
# Desde la raíz del proyecto
make model-test

# Equivalente manual:
cd tests && ../.venv/bin/pytest model/test_model.py -v
```

**Lo que prueba:**
- `test_model_preprocess_for_training`: Verifica que preprocess() retorna features con 10 columnas exactas y target con columna "delay"
- `test_model_preprocess_for_serving`: Verifica que preprocess() funciona sin target_column
- `test_model_fit`: Entrena con todos los datos, valida con 33% split, verifica métricas
- `test_model_predict`: Verifica predict() retorna List[int] del tamaño correcto

### Tests de Integración (API)

```bash
make api-test

# Equivalente manual:
cd tests && ../.venv/bin/pytest api/test_api.py -v
```

**Lo que prueba:**
- `test_should_get_predict`: POST /predict con datos válidos → 200 + {"predict": [...]}
- `test_should_failed_unkown_column_1`: MES=13 → 400
- `test_should_failed_unkown_column_2`: TIPOVUELO="O" → 400
- `test_should_failed_unkown_column_3`: OPERA="Argentinas" → 400

### Stress Test

```bash
# Primero iniciar la API:
uvicorn challenge:application --host 0.0.0.0 --port 8000

# En otra terminal:
make stress-test

# Para probar contra URL desplegada:
make stress-test STRESS_URL=https://tu-api.cloud.run
```

**Lo que prueba:** 100 usuarios concurrentes durante 60 segundos.

### Verificar sintaxis

```bash
python -m py_compile challenge/model.py
python -m py_compile challenge/api.py
python -m py_compile challenge/__init__.py
```

---

## 5. Edge Cases Considerados

### Validación de Input

| Escenario | Input | Respuesta Esperada | Implementación |
|-----------|-------|-------------------|----------------|
| MES = 0 | `{"MES": 0}` | 400 | Validador pydantic: `value < 1 or value > 12` |
| MES = 13 | `{"MES": 13}` | 400 | Validador pydantic |
| MES = "tres" | `{"MES": "tres"}` | 400 | Type validation de pydantic |
| TIPOVUELO = "O" | `{"TIPOVUELO": "O"}` | 400 | Validador: solo "I" o "N" |
| TIPOVUELO = "" | `{"TIPOVUELO": ""}` | 400 | Validador |
| OPERA desconocida | `{"OPERA": "Argentinas"}` | 400 | Validador contra set dinámico |
| OPERA válida | `{"OPERA": "Grupo LATAM"}` | 200 | Validador pasa |
| Flights array vacío | `{"flights": []}` | 422 | FastAPI default (lista vacía es válida) |
| Campos faltantes | `{"flights": [{"MES": 3}]}` | 422 | Pydantic requiere todos los campos |
| Campos extra | `{"flights": [{"OPERA": "...", "TIPOVUELO": "N", "MES": 3, "extra": true}]}` | 422 | Pydantic reject extra fields |

### Preprocesamiento y Predicción

| Escenario | Comportamiento | Implementación |
|-----------|---------------|----------------|
| Features faltantes (not top 10) | Zero-fill automático | `reindex(columns=TOP_10, fill_value=0)` |
| predict() sin fit() | Auto-entrena con datos default | Lazy initialization en predict() |
| DataFrame vacío en predict() | Retorna lista vacía | `_model.predict()` maneja shape (0, 10) |
| NaN en datos de entrenamiento | Error controlado con logging | Except en fit(), loggea error |
| Model training falla | API arranca sin modelo, predict() auto-entrena | Manejo de excepción en startup |

### Infraestructura

| Escenario | Comportamiento | Implementación |
|-----------|---------------|----------------|
| data.csv no encontrado | API no arranca (fail-fast) | Path absoluto con `pathlib.Path(__file__)` |
| Puerto ocupado | Uvicorn muestra error | Manejo estándar de uvicorn |
| Memoria insuficiente | OOM killer del container | Cloud Run con --memory=1Gi |
| Cold start (0→1 instancia) | ~5s para entrenar modelo | Aceptable para Cloud Run |
| 100 usuarios concurrentes | Modelo thread-safe | sklearn .predict() es read-only |
| Timeout del request | Cloud Run timeout 60s | Configurable en deploy |

---

## 6. Estructura del Proyecto

```
challenge_MLE/
├── .github/workflows/
│   ├── ci.yml              # CI: tests + build en cada push
│   └── cd.yml              # CD: deploy a Cloud Run en push a main
├── challenge/
│   ├── __init__.py          # Exporta `app` como `application`
│   ├── api.py               # FastAPI: endpoints, validación, startup
│   ├── exploration.ipynb    # Notebook original del DS (no modificar)
│   └── model.py             # DelayModel: preprocess, fit, predict
├── data/
│   └── data.csv             # Dataset de entrenamiento
├── docs/
│   └── challenge.md         # Esta documentación
├── tests/
│   ├── api/
│   │   └── test_api.py      # Tests de integración de la API
│   ├── model/
│   │   └── test_model.py    # Tests unitarios del modelo
│   └── stress/
│       └── api_stress.py    # Locust script para stress test
├── workflows/
│   ├── ci.yml               # Template original (no modificar)
│   └── cd.yml               # Template original (no modificar)
├── .coveragerc              # Configuración de cobertura
├── .gitignore               # Archivos ignorados por git
├── Dockerfile               # Multi-stage build (python:3.9-slim)
├── Makefile                 # Comandos de build/test
├── pyproject.toml           # Configuración del proyecto Python
├── requirements-dev.txt     # Dependencias de desarrollo
├── requirements-test.txt    # Dependencias de testing
└── requirements.txt         # Dependencias de producción
```

### Dependencias

| Paquete | Versión | Propósito |
|---------|---------|-----------|
| fastapi | ~0.86.0 | Framework web |
| pydantic | ~1.10.2 | Validación de datos |
| uvicorn | ~0.15.0 | Servidor ASGI |
| numpy | ~1.22.4 | Computación numérica |
| pandas | ~1.3.5 | Manipulación de datos |
| scikit-learn | ~1.3.0 | LogisticRegression |
| locust | ~1.6 | Stress testing |
| pytest | ~6.2.5 | Testing |
| pytest-cov | ~2.12.1 | Cobertura |

---

## 7. CI/CD

### CI Pipeline (`.github/workflows/ci.yml`)

```yaml
Triggers:
  - push a: main, develop, feature/*
  - pull_request a: main

Jobs:
  1. test:
     - Python 3.9
     - Install: requirements.txt + requirements-test.txt
     - Run: make model-test (cd tests && pytest model/)
     - Run: make api-test (cd tests && pytest api/)

  2. build:
     - Docker build (no push)
     - Cache: GitHub Actions cache
```

### CD Pipeline (`.github/workflows/cd.yml`)

```yaml
Triggers:
  - push a: main

Jobs:
  1. deploy:
     - Auth: Workload Identity Federation (OIDC)
     - Build + Push: gcr.io/PROJECT/flight-delay-api:SHA
     - Deploy: Cloud Run (managed, 1 CPU, 1Gi RAM)
     - Smoke test: GET /health (retry hasta 2 min)
     - Stress test opcional: locust (50 users, 30s)
```

### Secrets requeridos (GitHub → Settings → Secrets)

| Secret | Descripción |
|--------|-------------|
| `GCP_PROJECT_ID` | ID del proyecto GCP |
| `WORKLOAD_IDENTITY_PROVIDER` | Provider OIDC |
| `SERVICE_ACCOUNT_EMAIL` | Service account para deploy |
| `GCP_REGION` | Región (default: us-central1) |

---

## 8. Deploy en la Nube

### Usando Cloud Run (GCP — Recomendado)

```bash
# 1. Autenticar con GCP
gcloud auth login

# 2. Construir imagen
docker build -t gcr.io/$PROJECT_ID/flight-delay-api:latest .

# 3. Pushear a Container Registry
docker push gcr.io/$PROJECT_ID/flight-delay-api:latest

# 4. Desplegar en Cloud Run
gcloud run deploy flight-delay-api \
    --image gcr.io/$PROJECT_ID/flight-delay-api:latest \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --cpu=1 \
    --memory=1Gi \
    --min-instances=0 \
    --max-instances=10 \
    --concurrency=80 \
    --timeout=60s
```

### Verificar deploy

```bash
# Obtener URL
URL=$(gcloud run services describe flight-delay-api \
    --region us-central1 \
    --format='value(status.url)')

# Health check
curl -f $URL/health

# Predicción de ejemplo
curl -X POST $URL/predict \
    -H "Content-Type: application/json" \
    -d '{"flights": [{"OPERA": "Grupo LATAM", "TIPOVUELO": "N", "MES": 3}]}'

# Stress test
make stress-test STRESS_URL=$URL
```

### Actualizar Makefile

Actualizar la línea 26 con la URL del API desplegada:

```makefile
STRESS_URL = https://flight-delay-api-xxxxx-uc.a.run.app
```

---

## Apéndice: Extracto del Classification Report

El modelo entrenado con Logistic Regression + top 10 features + class balancing produce estas métricas (contra test split 33%):

```
              precision    recall  f1-score   support

           0       0.88      0.48      0.62     18073
           1       0.23      0.72      0.35      4425

    accuracy                           0.52     22498
   macro avg       0.56      0.60      0.49     22498
weighted avg       0.75      0.52      0.57     22498
```

- **Recall clase "1" (delay):** > 0.60 ✅ (umbral del test)
- **F1 clase "1":** > 0.30 ✅ (umbral del test)
- **Recall clase "0":** < 0.60 ✅ (umbral del test — el modelo deliberadamente sesgado a predecir delays)
- **F1 clase "0":** < 0.70 ✅ (umbral del test)

Esto cumple con todas las aserciones de `test_model_fit`.
