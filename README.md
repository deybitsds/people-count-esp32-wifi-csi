# Device-Free People Counting with Wi-Fi CSI on ESP32

Two ESP32 microcontrollers form a transmitter-receiver pair that captures Wi-Fi Channel State Information (CSI) across an indoor space. The CSI data feeds into a dual-path inference system: a Random Forest classifier trained in a Jupyter notebook, and a production RAG pipeline using ChromaDB vector search with a Gemini 2.5 Flash LLM via OpenRouter. A real-time web dashboard visualizes predictions.

## Architecture

The system splits across three layers: physical capture, inference, and presentation.

```
ESP32 (TX, AP mode) --> 802.11n packets --> ESP32 (RX, STA mode)
                                                  |
                                           Serial (921600 baud)
                                                  |
                                         bridge_csi.py
                                              |
                                      HTTP POST (JSON)
                                              |
                                    Django REST API
                                    /recognize/api/
                                    |              |
                              ChromaDB         Gemini 2.5 Flash
                              (BGE-M3)         (via OpenRouter)
                                    |              |
                                    SQLite (history)
                                              |
                                    toy_front dashboard
                                    (JS polling /api/historial/)
```

The ESP32 transmitter runs in Access Point mode broadcasting null-data packets. The receiver, configured as a Station, captures CSI for each received packet and prints it to the serial console at 921600 baud. The Python bridge reads this serial stream, batches packets, and forwards them as HTTP POST requests to the Django API.

Two inference paths coexist. The Random Forest path is trained offline in the Jupyter notebook and yields three classifier granularities. The RAG path embeds each incoming CSI signal using BGE-M3, retrieves the five most similar historical signals from ChromaDB, and passes them as few-shot context to Gemini 2.5 Flash, which returns a JSON prediction of 0--7 people.

## Hardware Requirements

You need two ESP32-WROOM-32 development boards. One is configured as the transmitter (TX), the other as the receiver (RX). The boards must be separated by at least 1--1.5 meters to ensure clear channel differentiation. Both connect to the development machine via USB for serial communication and firmware flashing.

Recommended operating band is 2.4 GHz, channel 6, with HT20 bandwidth. The Espressif ESP-IDF CSI implementation provides 64 subcarriers of I/Q data per packet for HT20 mode (128 I/Q values per packet after the 12-byte header is stripped).

## Firmware: TX and RX

The firmware lives in the `material/esp-csi/` submodule, sourced from the [Espressif ESP-CSI](https://github.com/espressif/esp-csi) repository. Two example projects are relevant: `csi_send` for the transmitter and `csi_recv` for the receiver.

### Transmitter (csi_send)

The TX firmware configures the ESP32 as a soft-AP broadcasting null-data packets. These packets carry no payload but trigger CSI feedback on the receiver side. The default channel and bandwidth settings in the firmware should match what the receiver expects.

Flash and monitor the TX on `/dev/ttyUSB0`:

```bash
source ~/esp/esp-idf/export.sh
cd material/esp-csi/examples/get-started/csi_send
idf.py -p /dev/ttyUSB0 flash monitor
```

After flashing, the TX continuously transmits. No further interaction is required.

### Receiver (csi_recv)

The RX firmware connects to the TX's AP and captures CSI for every received null-data packet. Each captured CSI frame is printed to the serial console as a structured line prefixed with `CSI_DATA`. The line format mirrors the CSV schema used throughout the project.

```bash
source ~/esp/esp-idf/export.sh
cd material/esp-csi/examples/get-started/csi_recv
idf.py -p /dev/ttyUSB1 flash monitor
```

The RX outputs data at roughly 100 packets per second, depending on the configured transmission interval and channel conditions.

## CSI Data Format

Each line from the receiver serial console is a CSV-formatted record. The columns, in order, are:

| Column | Description | Example |
|---|---|---|
| `type` | Always `CSI_DATA` | CSI_DATA |
| `role` | Station or Access Point | STA |
| `mac` | Source MAC address | E0:8C:FE:5C:56:F1 |
| `rssi` | Received Signal Strength Indicator (dBm) | -77 |
| `rate` | Physical layer rate | 11 |
| `sig_mode` | Signal mode (0=legacy, 1=HT) | 1 |
| `mcs` | Modulation and Coding Scheme | 1 |
| `bandwidth` | 0=HT20, 1=HT40 | 1 |
| `smoothing` | Channel smoothing enabled | 1 |
| `not_sounding` | Not sounding flag | 1 |
| `aggregation` | A-MPDU aggregation | 0 |
| `stbc` | Space-Time Block Coding | 0 |
| `fec_coding` | FEC coding type | 0 |
| `sgi` | Short Guard Interval | 0 |
| `noise_floor` | Noise floor estimate (dBm) | -97 |
| `ampdu_cnt` | A-MPDU count | 0 |
| `channel` | Primary channel | 6 |
| `secondary_channel` | Secondary channel position | 1 |
| `local_timestamp` | ESP32 local timer value | 1893708560 |
| `ant` | Antenna index | 0 |
| `sig_len` | Signal length | 110 |
| `rx_state` | Receiver state | 0 |
| `real_time_set` | Real time set flag | 0 |
| `real_timestamp` | Real timestamp (seconds) | 1894.09 |
| `len` | CSI data length (128 or 384) | 384 |
| `CSI_DATA` | I/Q values as space-separated signed integers in brackets | [110 96 6 0 ...] |
| `n_personas` | Label: number of people (only present in training CSVs) | 0 |

The `CSI_DATA` column requires explanation. For HT20 mode with 64 subcarriers, the raw data contains 384 bytes: a 12-byte header followed by 128 I/Q values per subcarrier (64 I + 64 Q), each stored as a 3-byte signed integer in the ESP32's internal format. However, when printed to serial, these values are decoded into standard signed integers. The length varies: a `len` of 128 indicates legacy mode (32 subcarriers), while 384 indicates HT20 (64 subcarriers). In practice, all CSV files in the dataset contain exactly 128 numeric values after the initial bracket, regardless of the `len` field, because the ESP32 firmware normalizes the output.

The file naming convention is `csi_p{N}_s{S}_YYYYMMDD_HHMMSS.csv`, where `N` is the number of people (0--7) and `S` is the session number (1 or 2). Each session represents a separate capture run with the same physical configuration.

## Offline ML Training: Random Forest Classifiers

The Jupyter notebook `csi_people_counting.ipynb` implements the complete training pipeline for three Random Forest classifiers. The notebook is self-contained and produces the models stored in `modelo/`.

### Feature Engineering Pipeline

The notebook reads each CSV, extracts the raw I/Q values from the `CSI_DATA` column, and converts them to amplitude using the Euclidean norm of each I/Q pair:

```python
def iq_to_amp(arr, start=CSI_START_IDX):
    iq = arr[start:]
    n = len(iq) // 2
    return np.sqrt(iq[0::2].astype(np.float32)**2 + iq[1::2].astype(np.float32)**2)
```

Amplitude-based features are more stable for classification than raw I/Q values, which vary with phase offsets introduced by hardware and environmental reflections.

The amplitude matrices are segmented using a sliding window approach with `WINDOW_SIZE=20` packets and `OVERLAP=0.5` (50% overlap). This yields multiple training samples per capture session, each covering approximately 200ms of channel activity.

For each window, the feature vector contains per-subcarrier statistics (mean, standard deviation, variance, 25th and 75th percentiles, peak-to-peak amplitude) plus global window statistics and meta-features from RSSI and noise floor. The total feature dimensionality is 6 per subcarrier × 64 subcarriers + 4 global features + 4 meta features = 392 features per window.

### Train/Test Split and Scaling

The notebook splits data by session: session 1 files (`_s1_`) are used for training, session 2 files (`_s2_`) for testing. This is a temporal split that evaluates generalization across different capture runs, which is more realistic than a random split that might mix correlated samples from the same session.

```python
(X_train if '_s1_' in f else X_test).append(feats)
(y_train if '_s1_' in f else y_test).append(lbls_arr)
```

A `StandardScaler` is fit on the training set and applied to both train and test sets. The scaler is saved alongside the models for use in any deployment scenario.

### Classifier Hierarchy

Three classifiers target different granularities of occupancy detection:

**Binary (rf_binario.pkl)**: Distinguishes empty (0) from occupied (1--7). Trained with 200 trees, max depth 15. This is the most reliable classifier, achieving approximately 92% accuracy and an F1 score of 0.95 on the session 2 test set. The confusion matrix shows balanced performance across both classes.

**3-class (rf_3clases.pkl)**: Maps labels to empty (0), low occupancy (1--3), and high occupancy (4--7). Uses 200 trees, max depth 18. Accuracy reaches approximately 67% with an F1-macro of 0.63. Most confusion occurs between the low and high occupancy groups, which is expected given the continuous nature of signal attenuation as people enter a space.

**8-class (rf_8clases.pkl)**: Predicts the exact count from 0 to 7 people. Uses 300 trees, max depth 22. Accuracy is 32% with an F1-macro of 0.27. The normalized confusion matrix reveals that misclassifications are concentrated in adjacent class bins -- the model rarely mistakes 2 people for 7, but frequently confuses 3 and 4. This pattern is consistent with the limited feature separation at fine granularities.

### Model Persistence

All models and the scaler are serialized with `joblib` to `modelo/`:

```
modelo/
  rf_binario.pkl
  rf_3clases.pkl
  rf_8clases.pkl
  scaler.pkl
```

These files are also duplicated in `api_rest_csi-signal/reconocer/modelo/` for the Django app to access locally, though the RAG pipeline does not use them directly.

## Production RAG Pipeline: Django + ChromaDB + Gemini

The Django REST API in `api_rest_csi-signal/` implements a retrieval-augmented generation pipeline for real-time inference. This approach replaces the Random Forest with a method that compares incoming CSI signals against the labeled historical database using vector similarity, then delegates the final classification to an LLM.

### Project Structure

```
api_rest_csi-signal/
  config/
    __init__.py
    env.py                          # environ.Env() + BASE_DIR singleton
    urls.py                         # Root URL routing
    wsgi.py / asgi.py               # Entry points
    django/
      base.py                       # Shared settings (INSTALLED_APPS, MIDDLEWARE, etc.)
      local.py                      # DEBUG=True, SQLite, CORS open
      production.py                 # DEBUG=False, env-based hosts
      test.py
  reconocer/
    __init__.py
    admin.py
    apps.py                         # ReconocerConfig
    models.py                       # CSIReading
    serializer.py                   # CSIReceiveSerializer
    views.py                        # RecognizeViewSet (core RAG logic)
    urls.py
    setup/setup.py                  # LLM initialization (legacy)
    prompts/prompt.py               # Invoice prompt (leftover, unused)
    datos/                          # CSV copies for vectorstore ingestion
    modelo/                         # ML model copies + bge-m3/
    vectorstore/chroma_db/          # Persisted ChromaDB index
    migrations/
  toy_front/
    __init__.py
    views.py                        # index() + historial_json()
    templates/toy_front/index.html  # Dashboard
  manage.py
  .env                              # API_KEY_OPEN_ROUTER, BASE_URL_OPEN_ROUTER
  db.sqlite3
```

The `config/` package contains only settings and WSGI/ASGI entry points. No business logic resides there. Settings are split across `base.py` for shared configuration and environment-specific overrides in `local.py`, `production.py`, and `test.py`. The `DJANGO_SETTINGS_MODULE` environment variable selects which settings file to use, defaulting to `config.django.local`.

### Data Model

The `CSIReading` model stores each prediction generated by the system:

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

`csi_raw` stores the raw integer array as a JSON-serialized string, not as a Django `JSONField`. This decision keeps the schema simple and avoids PostgreSQL-specific field types since the project uses SQLite. The `rssi` and `noise_floor` fields are nullable because the bridge can produce records where these values are unknown, though in practice they are always populated. `modelo_usado` defaults to `'rf_3clases'` for historical reasons; the RAG pipeline writes `'rag_gemini_flash'`.

### Input Validation

The serializer validates incoming requests. It expects a list of integers for the CSI data and optional RSSI/noise_floor values:

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

The `ListField(child=IntegerField())` approach means DRF's parser handles the JSON array conversion automatically. The validated data contains a native Python list, not a string. The serializer does not extend `ModelSerializer` because there is no one-to-one mapping between the incoming API fields and the model fields -- `csi_values` is transformed before storage.

### RAG Initialization and Thread Safety

The `RecognizeViewSet` initializes heavy components -- the HuggingFace BGE-M3 embedding model, the ChromaDB vector store, and the Gemini LLM -- lazily on the first request. This avoids blocking the Django startup process, which can take several seconds while BGE-M3 loads.

```python
class RecognizeViewSet(ViewSet):
    _vector_db = None
    _embeddings = None
    _llm = None
    _chain = None
    _initialization_lock = threading.Lock()
    _is_initialized = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not RecognizeViewSet._is_initialized:
            threading.Thread(target=self._initialize_components, daemon=True).start()
```

The initialization runs in a daemon thread with a threading lock to prevent duplicate initialization under concurrent requests. The `_is_initialized` flag is checked in each endpoint; if the system is still loading, it returns HTTP 503 Service Unavailable with a message to retry.

Inside `_initialize_components`, three operations occur sequentially:

1. **Embeddings model**: `HuggingFaceEmbeddings` loads the BGE-M3 model from `modelo/bge-m3/`. The model runs on CPU.
2. **Vector store**: If no persisted ChromaDB exists at `vectorstore/chroma_db/`, the system ingests all CSV files in `datos/`, converting each row's CSI data into a text document and storing it with metadata (person count, RSSI, noise floor). The ingestion splits documents into batches of 16 for memory efficiency. If a vector store already exists, it loads it directly.
3. **LLM chain**: A `ChatOpenAI` instance connects to Gemini 2.5 Flash via OpenRouter with temperature 0, 8 retries, and a 45-second timeout. The prompt template is constructed using LangChain's `ChatPromptTemplate`:

```python
prompt_template = ChatPromptTemplate.from_messages([
    (
        "user",
        """Eres un experto en análisis de señales WiFi CSI (Channel State Information) para determinar cuántas personas hay en una habitación.

Señal actual (CSI crudo - valores I/Q):
{senial_actual}

A continuación tienes las 5 señales más similares encontradas en la base de datos, con su cantidad de personas conocida:

{contexto}

Basándote en la similitud entre la señal actual y las señales de referencia, infiere CUÁNTAS PERSONAS hay exactamente (un número entero entre 0 y 7).

Responde ÚNICAMENTE con un objeto JSON en este formato exacto, sin texto adicional:
{{"cantidad_personas": <int 0-7>}}"""
    )
])
```

The chain is a LangChain expression language (LCEL) pipeline: `prompt_template | llm | StrOutputParser()`.

### Inference Flow

When the `recognize` endpoint receives a POST request:

1. The serializer validates the incoming JSON. Required field: `csi_values` (list of integers). Optional fields: `rssi`, `noise_floor` (default -100).
2. The CSI integer list is flattened to a space-separated string of numbers via `_csi_to_text`.
3. The string is used to query ChromaDB with `similarity_search_with_score(csi_text, k=5)`, returning the five most similar historical CSI signals and their cosine distances.
4. The retrieved documents and their distances, along with their metadata (n_personas, RSSI, noise_floor), are formatted into the `contexto` variable.
5. The chain is invoked with `senial_actual` (first 500 characters of the query) and `contexto` (formatted reference signals).
6. The LLM response is parsed with a regex for `"cantidad_personas": <number>`. The number is clamped to the 0--7 range.
7. A `CSIReading` record is created in SQLite with the prediction, and the response returns `{"cantidad_personas": N, "status": 200}`.

### Historial Endpoint

A second GET endpoint at `/recognize/historial/` returns the 50 most recent predictions as a JSON array. This endpoint is consumed by the dashboard for polling.

## Serial Bridge: bridge_csi.py

The bridge is a standalone Python script that connects the physical ESP32 receiver to the Django API. It reads raw serial output, parses CSI_DATA lines, and sends them as batched HTTP POST requests.

```bash
python bridge_csi.py /dev/ttyUSB1 --server http://localhost:8000
```

### Command-Line Arguments

| Argument | Default | Description |
|---|---|---|
| `port` | (required) | Serial port of the ESP32 RX |
| `--baud` | 921600 | Baud rate for serial connection |
| `--server` | http://34.172.215.14:8691 | Target server URL |
| `--endpoint` | /recognize/api/ | API endpoint path |
| `--interval` | 1.0 | Minimum interval between sends (seconds) |
| `--batch` | 5 | Packets to accumulate before sending |

### Parsing Logic

The `parse_csi_line` function treats each serial line as a CSV row and extracts the RSSI (column index 3), noise floor (column index 14), and the last column containing bracketed CSI integer data. Regex extraction of integers from the bracket handles variable whitespace and negative signs:

```python
def parse_csi_line(line):
    if 'CSI_DATA' not in line:
        return None
    reader = csv.reader(StringIO(line))
    row = next(reader)
    rssi = int(row[3])
    noise_floor = int(row[14])
    csi_str = row[-1]
    values = [int(x) for x in re.findall(r'-?\d+', csi_str)]
    return {'csi_values': values, 'rssi': rssi, 'noise_floor': noise_floor}
```

If parsing fails for a specific field, defaults of -100 are used for RSSI and noise floor, and the line is silently skipped if no CSI values can be extracted.

### Batching and Transmission

The bridge accumulates parsed packets in a deque. A transmission occurs when either the batch size is reached (default 5 packets) or the interval timer expires (default 1 second), whichever comes first. For each transmission, the RSSI and noise floor are averaged across all buffered packets, and the CSI values from the most recent packet are used as the payload.

```python
avg_rssi = sum(p['rssi'] for p in buffer) // len(buffer)
avg_nf = sum(p['noise_floor'] for p in buffer) // len(buffer)
latest = buffer[-1]
payload = {'csi_values': latest['csi_values'], 'rssi': avg_rssi, 'noise_floor': avg_nf}
```

The `buffer.clear()` call after each successful transmission discards accumulated packets, ensuring each CSI reading is used only once.

## Web Dashboard

The dashboard at the root URL `/` is a single HTML page (`toy_front/templates/toy_front/index.html`) with embedded CSS and JavaScript. It polls `GET /api/historial/` every 3 seconds and updates the view.

### Layout and Behavior

The dashboard has four card sections:

**Current Status** displays a large numeric readout of the latest prediction. The color changes based on occupancy: black for empty (0), amber for moderate (1--3), and red for high (4--7). A subtitle provides descriptive text ("No se detectan personas" or "Se detectan exactamente N personas").

**Signal** shows the RSSI value from the latest reading along with a gradient-filled bar that normalizes RSSI from -100 dBm to -50 dBm into a 0--100% range. This normalization is a heuristic -- RSSI values below -100 dBm are clamped to 0%, and values above -50 dBm to 100%.

**Summary** displays aggregate statistics for the current 100-record window: total readings, count of empty readings, and count of occupied readings.

**Historial** shows a table with timestamp, RSSI (with mini bar), and a color-coded badge for each prediction. The table auto-refreshes, and the empty state shows instructions for sending data.

The JavaScript uses `setInterval(fetchHistorial, 3000)` with `fetch` and renders data using template literals. No framework or build step is involved -- the dashboard depends only on the browser's native `fetch` API and ES6 features.

## Running the Full System

### Prerequisites

Install Python dependencies:

```bash
pip install django djangorestframework django-cors-headers django-environ python-dotenv \
            langchain-chroma langchain-huggingface langchain-openai langchain-core \
            pyserial requests numpy pandas scikit-learn joblib
```

The BGE-M3 embedding model must be downloaded to `api_rest_csi-signal/reconocer/modelo/bge-m3/`. This is a large model (~2.2 GB). The API attempts to load it on startup and will print an error if the path does not exist -- ChromaDB can use a default embedding function as a fallback, but prediction quality degrades.

Set up the `.env` file in `api_rest_csi-signal/`:

```
API_KEY_OPEN_ROUTER=sk-or-v1-your-key-here
BASE_URL_OPEN_ROUTER=https://openrouter.ai/api/v1
```

### Starting the API

```bash
cd api_rest_csi-signal

# First run: apply migrations
python manage.py migrate

# Start the development server
python manage.py runserver 0.0.0.0:8000
```

The first request to `/recognize/api/` will trigger the RAG initialization, which loads BGE-M3 and builds the ChromaDB index from the CSV files. This takes 30--60 seconds depending on the machine. Subsequent requests use the cached components.

### Starting the Bridge

On a machine with the ESP32 receiver connected:

```bash
python bridge_csi.py /dev/ttyUSB1 --server http://localhost:8000
```

Replace `/dev/ttyUSB1` with the correct serial port for your receiver. The TX transmitter should have been flashed earlier and be powered on.

### Accessing the Dashboard

Open `http://localhost:8000` in a browser after the API server starts. The dashboard will show "Esperando datos..." until the first CSI reading is processed. Once the bridge sends data, the dashboard updates every 3 seconds.

## Model Results

The Random Forest classifiers were evaluated on session 2 data after training on session 1:

| Classifier | Accuracy | F1 (macro) | Practical Use |
|---|---|---|---|
| Binary (empty/occupied) | 92.23% | 0.9535 | Presence detection |
| 3-class (0 / 1-3 / 4-7) | 66.62% | 0.6289 | Zoned occupancy (HVAC, lighting) |
| 8-class (0--7) | 31.74% | 0.2658 | Exact counting (limited) |

The binary detector is reliable enough for production presence sensing. The 3-class model provides coarse occupancy zoning. The 8-class model demonstrates the difficulty of fine-grained CSI-based counting with a single antenna pair and traditional machine learning. Adjacent-class errors dominate the confusion matrix -- the model tends to predict within one person of the ground truth, which suggests the limiting factor is feature resolution rather than classifier capacity.

The RAG pipeline's accuracy depends on the quality of the vector database and the LLM's ability to interpret the few-shot context. Early testing indicates that the LLM produces plausible predictions but systematic benchmarking against the held-out session 2 data has not been completed. The RAG pipeline is more flexible than the Random Forest because adding new labeled data only requires dropping CSVs into `datos/` and triggering a re-ingest.

## Configuration Points

The Django settings are split by environment. The local development configuration (`config/django/local.py`) enables DEBUG, allows all hosts, and opens CORS to all origins. The production configuration (`config/django/production.py`) disables DEBUG and reads allowed hosts and CORS origins from environment variables.

The `DJANGO_SETTINGS_MODULE` environment variable controls which settings file is loaded. The default is `config.django.local`. For production, set `DJANGO_SETTINGS_MODULE=config.django.production`.

The bridge script's batching parameters (--batch, --interval) control the trade-off between API load and prediction latency. A batch of 5 with a 1-second interval produces one prediction per second on average. Increasing the batch size smooths out RSSI variations but reduces responsiveness. Decreasing the interval below 0.5 seconds may overwhelm the API if the RAG pipeline's latency exceeds the incoming rate.

## Potential Improvements

The current system uses only amplitude features for the ML pipeline and raw I/Q values for the RAG pipeline. Phase-based features, such as the unwrapped phase difference across subcarriers or the covariance matrix of the channel response, could separate occupancy levels more clearly. The research papers in `material/` describe proportional fair algorithms and differential CSI approaches that extract cleaner occupancy signatures.

The 8-class classifier's performance is the primary bottleneck. A convolutional neural network operating on the raw CSI time-frequency matrix would capture spatial correlation across subcarriers and temporal patterns that the hand-engineered features miss. The notebook's heatmap visualization shows clear structural differences between occupancy levels that a CNN could exploit.

The RAG pipeline retrieves the five most similar vectors by cosine distance on raw I/Q text. This assumes that textual similarity of integer sequences correlates with occupancy similarity, which is a coarse approximation. A dedicated embedding model fine-tuned on CSI signals would produce more semantically meaningful vectors, and a hybrid search combining vector similarity with RSSI range filtering would reduce false retrievals.

The serial bridge is a single point of failure and limits the receiver to USB tethering distance from the server. Deploying the bridge on a dedicated gateway device (a Raspberry Pi connected to the ESP32) and transmitting over MQTT or WebSocket would decouple the capture hardware from the inference server.
