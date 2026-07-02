import os
import re
import json
import traceback
import logging
import threading

from rest_framework.response import Response
from rest_framework import status
from rest_framework.viewsets import ViewSet
from rest_framework.permissions import AllowAny
from rest_framework.decorators import action

from .serializer import CSIReceiveSerializer
from .models import CSIReading

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))
VECTORSTORE_DIR = os.path.join(BASE, 'vectorstore', 'chroma_db')
MODELO_DIR = os.path.join(BASE, 'modelo', 'bge-m3')

api_key_llm = os.getenv("API_KEY_OPEN_ROUTER")
base_url_llm = os.getenv("BASE_URL_OPEN_ROUTER")

class RecognizeViewSet(ViewSet):
    permission_classes = [AllowAny]

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

    def _csi_text_from_row(self, row):
        val = row['CSI_DATA']
        if isinstance(val, str):
            numbers = re.findall(r'-?\d+', val)
            return ' '.join(numbers)
        return ''

    def _load_csvs_and_build_vectorstore(self):
        import glob as glob_module
        import pandas as pd

        print("Ingesta: cargando CSVs de datos/ ...")
        csv_files = sorted(glob_module.glob(os.path.join(BASE, 'datos', 'csi_p*.csv')))
        documents = []
        skipped = 0

        for fpath in csv_files:
            fname = os.path.basename(fpath)
            df = pd.read_csv(fpath)
            for _, row in df.iterrows():
                csi_text = self._csi_text_from_row(row)
                if not csi_text:
                    skipped += 1
                    continue
                n_personas = int(row['n_personas'])
                rssi = int(row['rssi']) if pd.notna(row['rssi']) else None
                nf = int(row['noise_floor']) if pd.notna(row['noise_floor']) else None
                documents.append(Document(
                    page_content=csi_text,
                    metadata={
                        'n_personas': n_personas,
                        'rssi': rssi,
                        'noise_floor': nf,
                        'source_file': fname,
                    }
                ))
            print(f"  {fname}: {len(df)} filas")

        print(f"Total documentos: {len(documents)}, omitidos: {skipped}")

        if not documents:
            print("ERROR: No hay documentos para indexar")
            return

        print("Ingesta: creando base de datos vectorial y descartando colección previa...")
        db = Chroma(
            persist_directory=VECTORSTORE_DIR,
            embedding_function=RecognizeViewSet._embeddings,
            collection_name="csi_signals",
        )
        db.delete_collection()
        db = Chroma(
            persist_directory=VECTORSTORE_DIR,
            embedding_function=RecognizeViewSet._embeddings,
            collection_name="csi_signals",
        )

        batch_size = 16
        total_batches = (len(documents) + batch_size - 1) // batch_size
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            db.add_documents(batch)
            print(f"  Batch {i // batch_size + 1}/{total_batches} agregado ({len(batch)} docs)")

        RecognizeViewSet._vector_db = db
        print("Ingesta completada")

    def _initialize_components(self):
        with RecognizeViewSet._initialization_lock:
            if RecognizeViewSet._is_initialized:
                return
            try:
                print("Inicializando componentes RAG...")

                RecognizeViewSet._embeddings = HuggingFaceEmbeddings(
                    model_name=MODELO_DIR,
                    model_kwargs={'device': 'cpu'}
                )

                if not os.path.exists(VECTORSTORE_DIR) or not os.listdir(VECTORSTORE_DIR):
                    print("Vectorstore no encontrado. Iniciando ingesta desde CSVs...")
                    self._load_csvs_and_build_vectorstore()
                else:
                    RecognizeViewSet._vector_db = Chroma(
                        persist_directory=VECTORSTORE_DIR,
                        embedding_function=RecognizeViewSet._embeddings,
                        collection_name="csi_signals"
                    )
                    print("Base de datos vectorial cargada")

                RecognizeViewSet._llm = ChatOpenAI(
                    model='google/gemini-2.5-flash',
                    temperature=0,
                    max_retries=8,
                    request_timeout=45,
                    api_key=api_key_llm,
                    base_url=base_url_llm,
                )

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

                RecognizeViewSet._chain = prompt_template | RecognizeViewSet._llm | StrOutputParser()

                RecognizeViewSet._is_initialized = True
                print("Componentes RAG inicializados correctamente")

            except Exception as e:
                traceback.print_exc()

    def _csi_to_text(self, csi_values):
        if isinstance(csi_values, str):
            numbers = re.findall(r'-?\d+', csi_values)
        elif isinstance(csi_values, list):
            numbers = [str(x) for x in csi_values]
        else:
            numbers = []
        return ' '.join(numbers)

    @action(detail=False, methods=['post'])
    def recognize(self, request):
        if not RecognizeViewSet._is_initialized:
            return Response(
                {"error": "Sistema inicializando, intente de nuevo en unos segundos"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        try:
            serializer = CSIReceiveSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

            csi_raw = serializer.validated_data['csi_values']
            rssi = serializer.validated_data.get('rssi', -100)
            nf = serializer.validated_data.get('noise_floor', -100)

            csi_text = self._csi_to_text(csi_raw)
            if not csi_text:
                return Response({"error": "CSI data vacío"}, status=400)

            resultados = RecognizeViewSet._vector_db.similarity_search_with_score(
                csi_text, k=5
            )

            print("VECTORES MÁS PARECIDOS:")
            for i, (doc, score) in enumerate(resultados, 1):
                n_personas = doc.metadata.get('n_personas', '?')
                print(f"Vector {i}")
                print("=" * 9)
                print(f"Cantidad personas: {n_personas}")
                print(f"Vector: {doc.page_content[:100]}...")
                print()

            contexto_str = "\n\n".join([
                f"Señal {i+1} (distancia: {score:.4f}):\n"
                f"  Cantidad de personas: {doc.metadata.get('n_personas', '?')}\n"
                f"  RSSI: {doc.metadata.get('rssi', 'N/A')}\n"
                f"  Valores CSI: {doc.page_content[:200]}"
                for i, (doc, score) in enumerate(resultados)
            ])

            raw_response = RecognizeViewSet._chain.invoke({
                "senial_actual": csi_text[:500],
                "contexto": contexto_str,
            })

            cantidad = 0
            match = re.search(r'"cantidad_personas"\s*:\s*(\d+)', raw_response)
            if match:
                cantidad = max(0, min(7, int(match.group(1))))

            CSIReading.objects.create(
                rssi=rssi, noise_floor=nf,
                csi_raw=json.dumps(csi_raw) if isinstance(csi_raw, list) else str(csi_raw),
                prediccion=cantidad, confianza=1.0,
                modelo_usado='rag_gemini_flash'
            )

            return Response({
                "cantidad_personas": cantidad,
                "status": 200,
            })

        except Exception as e:
            logger.exception("Error en recognize")
            return Response({"error": str(e)}, status=500)

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
