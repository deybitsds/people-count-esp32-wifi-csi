
<task>
  <context>
    El proyecto es una API REST en Django que recibe señales
    de dos ESP32 vía WiFi sensing y determina cuántas personas hay en una habitación.
    El enfoque actual (en reconocer/views.py) no funciona bien y se va a reemplazar por
    un pipeline basado en RAG.
  </context>

  <current_state>
    <file path="reconocer/views.py">
      Contiene la lógica actual de recepción de señales y predicción.
    </file>
    <file path="reconocer/example.py">
      Contiene una implementación de referencia de mi flujo RAG (vectorización, Chroma,
      llamada a LLM). Usar como guía de estilo y patrones, NO copiar literalmente si no aplica.
    </file>
    <folder path="reconocer/datos/">
      CSVs con mediciones de señal WiFi etiquetadas por cantidad de personas (0 a 7,
      confirmar si el 0 existe revisando los archivos).
    </folder>
  </current_state>

  <goal>
    Reemplazar el algoritmo de predicción actual por un pipeline RAG que:
    1. Preprocesa y vectoriza los CSVs de datos/ en ChromaDB. Si se necesita poner en otro formato diferente al de los csv y en un solo archivo realiza este preprocamiento una sola vez, en un script llamado preprocessing.py (solo si es necesario) y guarda el resultado con un nombre simple que veas conveniente
       de ingesta no en cada request).
    2. En cada request a la API REST:
       a. Recibe la señal cruda del ESP32.
       b. La vectoriza con el mismo método de embeddings usado en la ingesta.
       c. Busca los 5 vectores más similares en ChromaDB.
       d. Loguea el resultado con este formato exacto:
          <log_format><![CDATA[
VECTORES MÁS PARECIDOS:
Vector 1
=========
Cantidad personas: <...>
Vector: <...>

Vector 2
=========
Cantidad personas: <...>
Vector: <...>

... (hasta Vector 5)
          ]]></log_format>
       e. Envía un prompt a un LLM incluyendo los 5 vectores similares (con su cantidad de
          personas) y la señal actual, pidiendo que infiera la cantidad de personas.
       f. El LLM debe responder en JSON: {"cantidad_personas": <int>, ...campos opcionales
          solo si aportan valor}.
       g. Parsea ese JSON usando los parsers de salida estructurada de LangChain
          (ej. PydanticOutputParser o with_structured_output).
       h. Devuelve el resultado al frontend.
  </goal>

  <constraints>
    <constraint>No modificar el frontend salvo lo estrictamente necesario para consumir
    la nueva respuesta de la API. Nada de mejoras visuales ni refactors de UI.</constraint>
    <constraint>No reescribir partes del backend no relacionadas con este flujo sin avisar
    primero.</constraint>
    <constraint>Seguir el estilo/patrones ya usados en reconocer/example.py cuando aplique
    (cliente Chroma, forma de construir el prompt, etc.).</constraint>
  </constraints>

  <instructions_for_claude_code>
    1. Primero lee reconocer/views.py, reconocer/example.py y explora reconocer/datos/ (estructura de
       columnas, si existe la clase "0 personas").
    2. Resume en texto plano qué encontraste y cómo planeas mapear example.py al nuevo flujo.
