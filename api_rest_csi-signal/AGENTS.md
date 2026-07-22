# AGENTS.md — CSI Signal Recognition API

Fuente de verdad de sintaxis y arquitectura para agentes de IA. Todo código nuevo **debe** cumplir estas reglas. Ninguna excepción sin aprobación explícita.

---

## 1. Contexto General

| Capa | Tecnología |
|---|---|
| Framework | Django 6.0.4 |
| API | Django REST Framework (ViewSet + @action) |
| Entorno | python-dotenv + django-environ (`environ.Env`) |
| CORS | django-cors-headers |
| Vector DB | ChromaDB (langchain_chroma) + BAAI/bge-m3 (HuggingFace) |
| LLM | google/gemini-2.5-flash via OpenRouter (ChatOpenAI) |
| RAG | LangChain LCEL (ChatPromptTemplate, StrOutputParser) |
| BD principal | SQLite (db.sqlite3) |
| Procesamiento | pandas (lectura CSV) |
| Zona horaria | America/Lima |

Los settings se organizan en **archivos separados por entorno**: `base.py` (compartido) + `local.py` / `production.py` / `test.py`. La selección se hace vía la variable de entorno `DJANGO_SETTINGS_MODULE`.

---

## 2. Estructura de Carpetas

Siempre debes mantener esta jerarquía exacta:

```
api_rest_csi-signal/
├── config/                        # Project config (NO apps aquí)
│   ├── __init__.py
│   ├── env.py                     # Único lugar para environ.Env() y BASE_DIR
│   ├── urls.py                    # Root URLconf
│   ├── wsgi.py
│   ├── asgi.py
│   └── django/
│       ├── __init__.py
│       ├── base.py                # Settings compartidos (INSTALLED_APPS, MIDDLEWARE, etc.)
│       ├── local.py               # DEBUG=True, SQLite, CORS_ALLOW_ALL_ORIGINS
│       ├── production.py          # DEBUG=False, env.list() para ALLOWED_HOSTS y CORS
│       └── test.py                # Overrides para testing
├── <app_name>/                    # Nombre en snake_case, corto, español o inglés
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py                    # AppConfig con default_auto_field = 'django.db.models.BigAutoField'
│   ├── models.py
│   ├── serializer.py              # Nombre en singular (no serializers.py)
│   ├── views.py
│   ├── urls.py
│   ├── tests.py
│   ├── migrations/
│   │   ├── __init__.py
│   │   └── ####_initial.py
│   ├── setup/                     # Solo si hay init perezoso (LLM, etc.)
│   │   └── setup.py
│   ├── prompts/                   # Solo si hay templates de LangChain
│   │   └── prompt.py
│   ├── datos/                     # Data estática (CSVs, entrenamiento)
│   └── templates/
│       └── <app_name>/
│           └── *.html
├── manage.py
├── .env                           # Cargado en manage.py y base.py
├── .gitignore
└── AGENTS.md
```

Reglas:
- **Nunca** pongas settings en un solo archivo `settings.py`. Siempre usa `config/django/base.py` + por entorno.
- **Nunca** crees un archivo `serializers.py` (plural). Usa `serializer.py` (singular).
- **Nunca** pongas lógica de negocio en `config/`. `config/` es solo para configuración.
- Los apps van siempre en la raíz del proyecto (no dentro de `config/` ni de `apps/`).

---

## 3. Reglas Estrictas de Sintaxis

### 3.1 Convenciones de Nombres

| Elemento | Convención | Ejemplo |
|---|---|---|
| Apps | `snake_case`, corto (1 palabra ideal) | `reconocer`, `toy_front` |
| Models | `PascalCase` singular | `CSIReading` |
| Model fields | `snake_case`, español o inglés descriptivo | `prediccion`, `confianza`, `csi_raw`, `modelo_usado` |
| Serializers | `PascalCase` + sufijo `Serializer` | `CSIReceiveSerializer` |
| ViewSets | `PascalCase` + sufijo `ViewSet` | `RecognizeViewSet` |
| ViewSet actions | `snake_case` | `recognize`, `historial` |
| Function-based views | `snake_case` | `index`, `historial_json` |
| URL names | `snake_case`, sufijo `_api` para endpoints REST | `recognize_api`, `historial_api` |
| Clase singleton lazy | `_` + `snake_case` (class-level) | `_vector_db`, `_is_initialized` |
| Métodos privados | `_` + `snake_case` | `_csi_to_text`, `_initialize_components` |
| Constantes módulo | `UPPER_CASE` | `BASE`, `VECTORSTORE_DIR`, `MODELO_DIR` |
| AppConfig class | `PascalCase` + sufijo `Config` | `ReconocerConfig` |
| Permissions class-level | `PascalCase` dentro de lista | `permission_classes = [AllowAny]` |

### 3.2 Type Hinting y Docstrings

**Type hints:** Todo el código NUEVO **debe** incluir type hints en:
- Parámetros de función y método
- Valores de retorno (`-> None`, `-> Response`, etc.)
- Atributos de clase (`_vector_db: Chroma | None = None`)
- Variables de módulo (`BASE: str = os.path.dirname(...)`)

```python
# CORRECTO (nuevo código):
def _csi_to_text(self, csi_values: list[int] | str) -> str: ...

# El código existente NO usa type hints — no los modifiques sin autorización.
```

**Docstrings:**
- **Models**: Siempre debes incluir un docstring describiendo el propósito del modelo y sus campos.
- **ViewSets**: Siempre debes incluir un docstring describiendo el endpoint, método HTTP, input y output.
- **Funciones privadas complejas**: Opcional pero recomendado.
- **Funciones públicas triviales** (`def index(request)`): No requieren docstring.

```python
class CSIReading(models.Model):
    """Almacena cada predicción de conteo de personas generada por el sistema RAG."""
    ...
```

### 3.3 Manejo de Queries

- Usa `Model.objects.all()`, `.create()`, `.filter()`, `.order_by()` directamente. **No uses Managers personalizados** (el proyecto no los tiene).
- Para listar historial, **siempre** usa `.order_by('-timestamp')[:N]`.
- No hay relaciones ForeignKey/ManyToMany en el proyecto, por lo tanto **no uses `select_related()` ni `prefetch_related()`**.
- Para queries que devuelven JSON, transforma el queryset inline con list comprehension:
  ```python
  qs = CSIReading.objects.all().order_by('-timestamp')[:50]
  data = [{"id": r.id, "timestamp": r.timestamp.isoformat()} for r in qs]
  ```
- Para inserción, usa `Model.objects.create(campo=valor)` (no el patrón `obj = Model(); obj.save()`).

### 3.4 Manejo de Datos JSON

El campo `csi_raw` del modelo `CSIReading` almacena arrays de enteros CSI como texto plano (`TextField`).

| Operación | Código |
|---|---|
| Escribir (desde lista Python) | `csi_raw=json.dumps(lista_valores)` |
| Leer (a lista Python) | `csi_values = json.loads(r.csi_raw)` |
| Validar entrada JSON (DRF) | `ListField(child=IntegerField())` en el serializer |

Reglas:
- **Nunca** uses `JSONField` de Django — el proyecto usa `TextField` + serialización manual con `json.dumps()` / `json.loads()`.
- El serializer recibe `csi_values` como **lista nativa de Python** (DRF ya parsea el JSON del request). La conversión a string JSON ocurre solo al guardar en el modelo.
- Para exponer `csi_raw` en un GET, transforma el string a lista: `json.loads(r.csi_raw)`.

---

## 4. Patrones de Diseño Locales

### 4.1 ViewSet + @action (no ModelViewSet)

Siempre debes usar `rest_framework.viewsets.ViewSet` con `@action(detail=False, methods=['<method>'])` para cada endpoint. **Nunca uses `ModelViewSet` a menos que explícitamente se requiera CRUD completo.**

### 4.2 Thread-Safe Singleton Inicialización Perezosa

Para componentes pesados (LLM, vector DB, embeddings) usa este patrón exacto:

```python
import threading

class MyViewSet(ViewSet):
    _heavy_component = None
    _initialization_lock = threading.Lock()
    _is_initialized = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not type(self)._is_initialized:
            threading.Thread(target=self._initialize_components, daemon=True).start()
```

Reglas:
- Usa **class-level variables** (no instance-level) con **doble guion bajo** (`_`)
- Usa `threading.Lock()` para proteger la inicialización
- El hilo debe ser `daemon=True`
- La variable `_is_initialized` evita reintentar si falló
- Los endpoints deben verificar `_is_initialized` y devolver `503` si no está listo

### 4.3 Plain Serializer para Input Validation

Usa `serializers.Serializer` (no `ModelSerializer`) para validar datos de entrada de la API. El `ModelSerializer` solo se usa si necesitas serialización bidireccional automática de modelos completos.

### 4.4 Function-Based Views para Páginas Simples

Usa FBV (`def view_name(request)`) para:
- Renderizado de templates HTML
- Endpoints JSON triviales que solo leen datos y devuelven JsonResponse

### 4.5 Lógica de Negocio en Views (sin Services/Selectors)

El proyecto **no usa** la capa Services/Selectors. Toda la lógica de negocio vive directamente en `views.py`. Si creas nuevo código, **evalúa si la lógica merece separación**; si es directa y corta (< 30 líneas), déjala en views. Si crece, migra a `services.py`.

### 4.6 Settings Split + django-environ

Siempre debes:
- Definir `env = environ.Env()` y `BASE_DIR` en `config/env.py` (único lugar)
- Heredar de `base.py` con `from .base import *` en cada archivo de entorno
- Usar `env('VAR_NAME', default='valor')` para leer variables de entorno
- Cargar `.env` en `base.py` con `env.read_env()`

### 4.7 CORS: Local vs Producción

- **local.py**: `CORS_ALLOW_ALL_ORIGINS = True`
- **production.py**: `CORS_ALLOWED_ORIGINS = env.list("CORS_ALLOWED_ORIGINS")`

---

## 5. Ejemplos de Código (Templates)

### 5.1 Modelo

```python
from django.db import models


class CSIReading(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    rssi = models.IntegerField(null=True, blank=True)
    noise_floor = models.IntegerField(null=True, blank=True)
    csi_raw = models.TextField()
    prediccion = models.IntegerField(null=True, blank=True)
    confianza = models.FloatField(null=True, blank=True)
    modelo_usado = models.CharField(max_length=20, default='rf_3clases')
```

Reglas:
- **Nunca** agregues `Meta` class a menos que sea estrictamente necesario
- **Nunca** agregues `__str__` a menos que lo requiera el admin
- Los campos opcionales llevan `null=True, blank=True`
- `auto_now_add=True` para timestamps de creación
- `csi_raw` es `TextField` (no `JSONField`): escribe con `json.dumps(lista)`, lee con `json.loads(r.csi_raw)`

### 5.2 Serializer (Plain, no ModelSerializer)

```python
from rest_framework import serializers


class CSIReceiveSerializer(serializers.Serializer):
    csi_values = serializers.ListField(
        child=serializers.IntegerField(),
        allow_empty=False
    )
    rssi = serializers.IntegerField(required=False, default=-100)
    noise_floor = serializers.IntegerField(required=False, default=-100)
```

Reglas:
- **Nunca** uses `class Meta` en un Serializer plano
- Usa `required=False` + `default=` para campos opcionales
- Usa `allow_empty=False` para listas que no pueden llegar vacías
- `ListField(child=IntegerField())` valida que el JSON recibido sea un array de enteros; el `validated_data` ya contiene una **lista Python**, no un string

### 5.3 ViewSet con @action y Singleton Lazy

```python
import os
import threading
import re
import json

from rest_framework.response import Response
from rest_framework import status
from rest_framework.viewsets import ViewSet
from rest_framework.permissions import AllowAny
from rest_framework.decorators import action

from .serializer import CSIReceiveSerializer
from .models import CSIReading


class RecognizeViewSet(ViewSet):
    permission_classes = [AllowAny]

    _vector_db = None
    _is_initialized = False
    _initialization_lock = threading.Lock()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not RecognizeViewSet._is_initialized:
            threading.Thread(target=self._initialize_components, daemon=True).start()

    def _initialize_components(self) -> None:
        with RecognizeViewSet._initialization_lock:
            if RecognizeViewSet._is_initialized:
                return
            # Inicializar componentes pesados aquí
            RecognizeViewSet._is_initialized = True

    @action(detail=False, methods=['post'])
    def recognize(self, request):
        if not RecognizeViewSet._is_initialized:
            return Response(
                {"error": "Sistema inicializando, intente de nuevo en unos segundos"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        serializer = CSIReceiveSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        # ... lógica de negocio ...
        CSIReading.objects.create(
            rssi=serializer.validated_data.get('rssi', -100),
            noise_floor=serializer.validated_data.get('noise_floor', -100),
            csi_raw=json.dumps(serializer.validated_data['csi_values']),
            prediccion=cantidad,
            confianza=1.0,
            modelo_usado='rag_gemini_flash'
        )
        return Response({"cantidad_personas": cantidad, "status": 200})

    @action(detail=False, methods=['get'])
    def historial(self, request):
        qs = CSIReading.objects.all().order_by('-timestamp')[:50]
        data = [{
            "id": r.id,
            "timestamp": r.timestamp.isoformat(),
            "rssi": r.rssi,
            "prediccion": r.prediccion,
            "confianza": r.confianza,
        } for r in qs]
        return Response(data)
```

### 5.4 Function-Based View

```python
from django.shortcuts import render
from django.http import JsonResponse
from reconocer.models import CSIReading


def index(request):
    return render(request, 'toy_front/index.html')


def historial_json(request):
    qs = CSIReading.objects.all().order_by('-timestamp')[:100]
    data = [{
        "id": r.id,
        "ts": r.timestamp.isoformat(),
        "rssi": r.rssi,
        "pred": r.prediccion,
        "conf": r.confianza if r.confianza is not None else 0.0,
    } for r in qs]
    return JsonResponse(data, safe=False)
```

### 5.5 URL Patterns

**App-level (`<app>/urls.py`):**
```python
from django.urls import path
from .views import RecognizeViewSet

urlpatterns = [
    path('api/', RecognizeViewSet.as_view({'post': 'recognize'}), name='recognize_api'),
    path('historial/', RecognizeViewSet.as_view({'get': 'historial'}), name='historial_api'),
]
```

**Root (`config/urls.py`):**
```python
from django.contrib import admin
from django.urls import path, include
from toy_front.views import index, historial_json

urlpatterns = [
    path("admin/", admin.site.urls),
    path("recognize/", include('reconocer.urls')),
    path("", index, name='dashboard'),
    path("api/historial/", historial_json, name='historial_json'),
]
```

Reglas para URLs:
- **Nunca** uses `router.register()` ni `DefaultRouter` — siempre mapea manualmente con `ViewSet.as_view({'method': 'action'})`
- Las URLs de app se montan con `include('app_name.urls')`
- Los nombres de URL llevan sufijo `_api` para endpoints de datos y descriptivo para páginas

### 5.6 App Config

```python
from django.apps import AppConfig


class ReconocerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'reconocer'
```

### 5.7 Settings (base.py + local.py)

**`config/env.py`:**
```python
import environ
from pathlib import Path

env = environ.Env()
BASE_DIR = Path(__file__).resolve().parent.parent
```

**`config/django/base.py`:**
```python
from config.env import env, BASE_DIR
import os

env.read_env(os.path.join(BASE_DIR, '.env'))

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "corsheaders",
    "rest_framework",
    "reconocer",
    "toy_front",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]
WSGI_APPLICATION = "config.wsgi.application"
TIME_ZONE = "America/Lima"
USE_TZ = True
STATIC_URL = "static/"
```

**`config/django/local.py`:**
```python
from .base import *
from config.env import env

DEBUG = True
ALLOWED_HOSTS = ["*"]
CORS_ALLOW_ALL_ORIGINS = True
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}
```

**`config/django/production.py`:**
```python
from .base import *
from config.env import env

DEBUG = False
ALLOWED_HOSTS = env.list("ALLOWED_HOSTS")
CORS_ALLOWED_ORIGINS = env.list("CORS_ALLOWED_ORIGINS")
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}
```

### 5.8 manage.py y Entrypoints

**`manage.py`:**
```python
#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
from config.env import env, BASE_DIR

env.read_env(os.path.join(BASE_DIR, '.env'))

def main():
    os.environ.setdefault(
        'DJANGO_SETTINGS_MODULE',
        env('DJANGO_SETTINGS_MODULE', default='config.django.local'))
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(...) from exc
    execute_from_command_line(sys.argv)

if __name__ == '__main__':
    main()
```

**`config/wsgi.py` y `config/asgi.py`:**
```python
import os
from config.env import env
from django.core.wsgi import get_wsgi_application

os.environ.setdefault(
    'DJANGO_SETTINGS_MODULE',
    env('DJANGO_SETTINGS_MODULE', default='config.django.local'))
application = get_wsgi_application()
```

---

## Resumen de Prohibiciones

| Prohibición | Razón |
|---|---|
| No uses `ModelViewSet` | El proyecto usa `ViewSet` + `@action` explícito |
| No uses `DefaultRouter` | Las URLs se mapean a mano con `as_view()` |
| No uses `serializers.py` (plural) | El proyecto usa `serializer.py` (singular) |
| No agregues `Meta` class a modelos sin razón | El proyecto actual no usa Meta en sus modelos |
| No uses `select_related()` / `prefetch_related()` | No hay relaciones FK en el proyecto |
| No crees `services.py` / `selectors.py` sin justificación | El proyecto pone toda la lógica en views |
| No uses `settings.py` | Los settings van en `config/django/base.py` + por entorno |
| No muevas apps dentro de subdirectorios | Todas las apps están en la raíz del proyecto |
| No modifiques el estilo del código existente (sin type hints) | Solo el código NUEVO debe incluir type hints |
| No uses `obj = Model(); obj.field = val; obj.save()` | Siempre usa `Model.objects.create(field=val)` |
