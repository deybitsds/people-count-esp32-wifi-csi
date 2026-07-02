
from django.shortcuts import render

# Create your views here.
import traceback
from .serializer import ChatbotMessageSerializer
from rest_framework import status
from rest_framework.decorators import action
from rest_framework.response import Response
import adrf.viewsets as viewsets
from django.utils import timezone
import uuid

#model imports
from .models import Contabilidad, Conversacion, Mensaje

# chatbot imports
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableBranch, RunnableLambda
from langchain_chroma import Chroma
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
import pandas as pd

from rest_framework.permissions import AllowAny

import jsonlines
import threading
from dotenv import load_dotenv
import os

load_dotenv()
api_key_llm = os.getenv("API_KEY_OPEN_ROUTER")
base_url_llm = os.getenv("BASE_URL_OPEN_ROUTER")
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = "DSContChatBot_Produccion_Cloud"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

class ChatbotViewSet(viewsets.ModelViewSet):
    permission_classes = [AllowAny]
    serializer_class = ChatbotMessageSerializer

    # Class-level variables for singleton pattern
    # vector stores
    _manual_vector_db = None
    _video_vector_db = None

    #chains
    _general_llm_chain = None
    _image_chain = None
    _video_chain = None
    _history_aware_retriever_chain = None
    _input_chain = None
    _full_chain = None
    _parallel_retriever_chain_manual_video = None

    _initialization_lock = threading.Lock()
    _is_initialized = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize components in background on first instance creation
        # Note: In a production async environment (ASGI), be careful with threads in __init__.
        # Ideally, this logic should move to apps.py (AppConfig.ready), but it works here for now.
        print('🚀 ~ views.py ~ 45 ~ hello:')
        if not ChatbotViewSet._is_initialized:
            threading.Thread(target=self._initialize_components, daemon=True).start()

    def _initialize_components(self):
        """Initialize vector database and chain once in a thread-safe manner"""
        with ChatbotViewSet._initialization_lock:
            if ChatbotViewSet._is_initialized:
                return
            
            try:
                print("Initializing chatbot components...")
                
                # Configuration
                persist_directory = 'chatbot/vectorStore/chroma_db'
                
                # Initialize embeddings
                model_embeddings_rag = HuggingFaceEmbeddings(
                    model_name="BAAI/bge-m3",
                    model_kwargs={'device': 'cpu'}
                )
                model_embeddings_router = HuggingFaceEmbeddings(
                    model_name="BAAI/bge-m3",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )

                if not os.path.exists(persist_directory) or not os.listdir(persist_directory):

                    #+++++++++++++VECTORIZAR MANUAL+++++++++++++++++
                    print("creando base de datos del manual")
                    manual_file_path = "chatbot/documentation/manual_cleaned.md"

                    try:
                        with open(manual_file_path, 'r', encoding='utf-8') as f:
                            markdown_text = f.read()
                    except FileNotFoundError:
                        print(f"Error: The file '{manual_file_path}' was not found.")
                        markdown_text = ""

                    if markdown_text:

                        headers_to_split_on = [
                            ("#", "Header 1"),
                            ("##", "Header 2"),
                            ("###", "Header 3"),
                            ("####", "Header 3"),
                            ("#####", "Header 4"),
                        ]

                        manual_markdown_splitter = MarkdownHeaderTextSplitter(
                            headers_to_split_on=headers_to_split_on,
                            strip_headers=False
                        )

                        manual_splits = manual_markdown_splitter.split_text(markdown_text)
                        
                        ChatbotViewSet._manual_vector_db = Chroma.from_documents(
                            documents=manual_splits,
                            embedding=model_embeddings_rag,
                            persist_directory=persist_directory,
                            collection_name="manual_vector_db"
                        )
                        print(f"Base de datos vectorial de manual creada")

                    #+++++++++++++VECTORIZAR VIDEOS+++++++++++++++++
                    print("creando base de datos de videos")
                    video_docs = []
                    video_file_path = "chatbot/documentation/videos_docs.jsonl"
                    with jsonlines.open(video_file_path, mode='r') as reader:
                        for doc_dict in reader:
                            video_docs.append(Document(**doc_dict))

                    video_splitter = RecursiveCharacterTextSplitter(
                        chunk_size = 2000, chunk_overlap=500
                    )

                    video_splits = video_splitter.split_documents(video_docs)

                    ChatbotViewSet._video_vector_db = Chroma.from_documents(
                                            documents=video_splits,
                                            embedding=model_embeddings_rag,
                                            collection_name="video_vector_db",
                                            persist_directory=persist_directory
                                        )
                    print(f"Base de datos vectorial de videos creada")

                    #+++++++++++++VECTORIZAR Q&A+++++++++++++++++
                    print("creando base de datos de Q&A")
                    df_QandA = pd.read_excel("chatbot/documentation/Formato Q&A chatbot.xlsx")

                    QandA_documents = [
                        Document(
                            page_content=d,
                            metadata={
                                "respuesta": str(a) + str(b),
                                "modulo": c
                            }
                        )
                        for a, b, c, d in zip(
                            df_QandA['explicacion_del_problema'], 
                            df_QandA['solucion'], 
                            df_QandA['modulo'], 
                            df_QandA['page_content']
                        )
                    ]

                    QandA_splitter = CharacterTextSplitter(
                        chunk_size = 500, chunk_overlap = 0
                    )

                    QandA_splits = QandA_splitter.split_documents(QandA_documents)

                    ChatbotViewSet._QandA_vector_db = Chroma.from_documents(
                        documents=QandA_splits,
                        embedding=model_embeddings_rag,
                        collection_name="QandA_vector_db",
                        persist_directory=persist_directory
                    )

                    print(f"Base de datos vectorial de Q&A creada")

                    #+++++++++++++VECTORIZAR Routing+++++++++++++++++

                    error_examples = [
                        "Me aparece un mensaje de error al guardar",
                        "No puedo timbrar, sale error 302",
                        "El sistema está trabado en la carga",
                        "Se cerró inesperadamente",
                        "Me aparece el error 'list index'",
                        "No me puedo conectar a la base de datos",
                        "No me puedo conectar al servidor",
                        "Me aparece un error de usuario"
                        "Me aparece un error de inicio de sesión"
                    ]

                    manual_examples = [
                        "¿explicame que es un amarre configurar el sire api?",
                        "¿como configurar el sire api?",
                        "¿Que son las percepciones?",
                        "¿Cuáles son los pasos para crear una empresa?",
                        "Guía para configurar el catálogo de cuentas",
                        "¿Cómo doy de alta un activo fijo?",
                        "Instrucciones para cancelar una factura",
                        "¿Qué es una detracción en el sistema?",
                        "Definición de asiento contable",
                        "Explicación de póliza de diario vs egreso",
                        "¿A qué se refiere el concepto de depreciación acumulada?",
                        "Glosario de términos fiscales",
                        "¿Qué es el DIOT?",
                        "¿como hacer backups en el dscont?"
                        "¿como hago copias de seguridad?"
                    ]

                    docs = []
                    for text in error_examples:
                        docs.append(Document(page_content=text, metadata={"route": "ERROR_ROUTE"}))

                    for text in manual_examples:
                        docs.append(Document(page_content=text, metadata={"route": "MANUAL_ROUTE"}))

                    ChatbotViewSet._vector_db_routing = Chroma.from_documents(
                        documents=docs,
                        embedding=model_embeddings_router,
                        collection_name="routing_vector_db",
                        persist_directory=persist_directory
                    )

                else:
                    # Load existing vector database
                    print("Cargando base de datos vectorial del manual")
                    ChatbotViewSet._manual_vector_db = Chroma(
                        persist_directory=persist_directory,
                        embedding_function=model_embeddings_rag,
                        collection_name="manual_cloud_vector_db"
                    )
                    print("Bases de datos vectorial del manual cargada")

                    print("Cargando base de datos vectorial de los videos")
                    ChatbotViewSet._video_vector_db = Chroma(
                        persist_directory=persist_directory,
                        embedding_function=model_embeddings_rag,
                        collection_name="video_cloud_vector_db"
                    )
                    print("Bases de datos vectorial de los videos cargada")

                # Initialize LLM and chain
                if ChatbotViewSet._manual_vector_db and ChatbotViewSet._video_vector_db:
                    # modelos de lenguaje
                    llm = ChatOpenAI(
                        model='google/gemini-2.5-flash-lite',
                        temperature = 0,
                        max_retries=8, 
                        request_timeout=45,
                        api_key=api_key_llm,
                        base_url=base_url_llm,
                    )

                    # ==========================================================
                    # prompts auxiliares
                    # ==========================================================

                    template_imagen = """
                    Actúa como un Especialista de Soporte Técnico para el software contable DSCont.
                    Tu tarea es analizar la captura de pantalla proporcionada, información técnica detallada para formular una consulta en una base de datos vectorial.

                    Por favor, analiza la imagen y genera un informe estructurado con los siguientes campos, siempre que se tenga informacíon relevante de cada uno:

                    1. Ubicación Exacta (Ruta de Navegación):
                    - Identifica el módulo principal (ej. Contabilidad, Planillas, Activos Fijos).
                    - Identifica la pestaña, submenú o ventana específica activa.
                    - Transcribe el título de la ventana actual tal como aparece en la barra superior.

                    2. Análisis del Error (Crítico) (Si la imagen es de un error del sistema, destacalo):
                    - **Transcripción Literal:** Escribe el mensaje de error exacto (palabra por palabra) que aparece en cuadros de diálogo, barras de estado o tooltips.
                    - **Códigos de Error:** Extrae cualquier número o código alfanumérico (ej. "Error 339", "SQL-01").
                    - **Tipo de Alerta:** Identifica si es una advertencia, error crítico o información.

                    3. Contexto Operativo y Estado:
                    - Describe qué operación parece estar intentando realizar el usuario (ej. "Intentando importar un asiento", "Generando reporte de ventas").
                    - Identifica campos visibles relevantes (fechas, rucs, montos) que puedan ser la causa (ej. "El campo 'Periodo' está vacío").
                    - Nota si hay botones deshabilitados (grisáceos) que indiquen falta de permisos o pasos previos incompletos.

                    4. Palabras Clave para Indexación (Tags):
                    - Lista 5-7 palabras clave técnicas relacionadas con la pantalla (ej. "Asientos", "Importación", "Sunat", "Libros Electrónicos").


                    **Formato de Salida:**
                    SOLO dame una query para hacer una consulta en la base de datos vectorial, no añadas palabras como "busca en ..." solo la infomacion relevante de la imagen en texto plano, sin ningún formato en especial.
                    """

                    prompt_imagen = ChatPromptTemplate.from_messages([
                        (
                            "user", 
                            [
                                {"type": "text", "text": template_imagen},
                                {"type": "image_url", "image_url": {"url": "{image_uri}"}}
                            ]
                        )
                    ])

                    condense_q_template_text = """
                    Dada la conversación histórica y la entrada actual del usuario, tu tarea es formular una pregunta independiente y completa para un buscador.

                    Instrucciones:
                    1. La entrada actual consta de una "Pregunta" y opcionalmente un "Contexto Visual".
                    2. **Si hay Contexto Visual:** Úsalo para reemplazar referencias vagas en la pregunta (como "esto", "este error", "aquí") con los detalles técnicos específicos de la imagen.
                    3. **Si NO hay Contexto Visual (está vacío):** Ignóralo y reformula la pregunta basándote solo en el historial de chat.
                    4. **Si el usuario uso la palabra error, usa la palabra error siempre para reformular la pregunta.

                    Objetivo:
                    Generar una única frase de búsqueda clara y específica. NO respondas la pregunta.

                    ## Historial de Conversación
                    {chat_history}

                    --- ENTRADA ACTUAL DEL USUARIO ---
                    Contexto Visual (Opcional): {image_interpretation}
                    Pregunta: {pregunta_usuario}
                    ----------------------------------
                    Pregunta Reformulada:
                    """

                    condense_q_prompt = ChatPromptTemplate.from_messages([
                        (
                            "user", 
                            [
                                {
                                    "type": "text", 
                                    "text": condense_q_template_text
                                }
                            ]
                        )
                    ])

                    # ==========================================================
                    # prompts finales para el llm
                    # ==========================================================
                    general_template_without_image_manual_video = """
                    Eres YachAI, el asistente virtual de atención al cliente experto en el software contable DSCont.
                    Tu objetivo es responder las preguntas de los usuarios de forma clara, amable y conversacional.

                    ## Historial de Conversación
                    {chat_history}

                    ## Contexto Técnico Recuperado
                    Contexto del manual:
                    {manual_context}

                    Contexto de videos:
                    {video_context}
                    // Formato de videos: [{{titulo: "...", link: "..."}}, ...]

                    ## Pregunta del Usuario
                    {pregunta_usuario}

                    ## Reglas para Responder
                    1. **Saludos e Interacción:** Si la entrada del usuario es un saludo (ej. "hola"), agradecimiento o despedida, ignora el contexto técnico. Responde naturalmente como YachAI y ofrece tu ayuda.
                    2. **Respuesta Técnica:** Si la respuesta está en el Contexto del manual, responde basándote *únicamente* en ella.
                    3. **Información Inexistente (Honestidad):** Si la pregunta es clara pero la respuesta NO está en el manual ni videos, SÉ HONESTO. Di: "Entiendo tu consulta, pero mi base de conocimientos actual no contiene información específica sobre este tema. Te sugiero contactar a soporte humano de DSCont". NO inventes información.
                    4. **Ambigüedad:** Si la pregunta es vaga (ej. "tengo un error"), pide detalles amablemente (módulo, mensaje exacto) basándote en el historial si aplica.
                    5. **Videos:** Si diste una respuesta técnica (Regla 2) y hay un video relevante, agrégalo al final.
                    """

                    general_prompt_without_image_manual_video = ChatPromptTemplate.from_messages([
                        (
                            "user", 
                            [
                                {
                                    "type": "text", 
                                    "text": general_template_without_image_manual_video
                                }
                            ]
                        )
                    ])

                    general_template_with_image_manual_video = """
                    Eres YachAI, el asistente virtual de atención al cliente experto en el software contable DSCont.

                    ## Historial de Conversación
                    {chat_history}

                    ## Contexto Técnico Recuperado
                    Contexto del manual:
                    {manual_context}

                    Contexto de videos:
                    {video_context}

                    ## Pregunta del Usuario y Análisis de Imagen
                    Pregunta: {pregunta_usuario}
                    Resumen de Imagen: {image_interpretation}

                    ## Reglas para Responder
                    1. **Saludos:** Si es solo un saludo, responde naturalmente como YachAI.
                    2. **Respuesta Técnica:** Combina la pregunta y el resumen de la imagen. Si la solución está en el manual, responde usándolo.
                    3. **Información Inexistente (Honestidad):** Si la imagen muestra un error claro pero el manual NO lo menciona, admite tu limitación: "Veo el error en la imagen, pero no tengo documentación sobre este caso específico. Por favor contacta a soporte humano".
                    4. **Ambigüedad:** Si la imagen no es clara o la pregunta es vaga, pide más detalles.
                    5. **Videos:** Si hay video relevante para la solución, agrégalo.
                    """

                    general_prompt_with_image_manual_video = ChatPromptTemplate.from_messages([
                        (
                            "user", 
                            [
                                {
                                    "type": "text", 
                                    "text": general_template_with_image_manual_video
                                }
                            ]
                        )
                    ])

                    # Retrievers
                    video_retriever = ChatbotViewSet._video_vector_db.as_retriever()
                    manual_retriever = ChatbotViewSet._manual_vector_db.as_retriever()

                    # Funciones auxiliares

                    def check_for_uri(input: dict) -> bool:
                        return bool(input.get("image_uri"))

                    def return_video_link(docs):
                        return list({(doc.metadata['source'], doc.metadata['title']) for doc in docs})

                    def has_history(input: dict) -> bool:
                        return len(input.get("chat_history", [])) > 0

                    async def fetch_history_for_chain(input_dict, limit = 10):
                        idConversacion = input_dict.get("idConversacion")

                        # Esta consulta sigue intacta porque la relación entre Mensaje y Conversacion no cambió
                        messages_data = Mensaje.objects.filter(
                            conversacion__id_conversacion=idConversacion
                        ).order_by('-fecha_envio').values('contenido_mensaje', 'chatbot_mensaje')[:limit]

                        history_buffer = []

                        async for data in messages_data:
                            if data['chatbot_mensaje']:
                                history_buffer.append(AIMessage(content=data['contenido_mensaje']))
                            else:
                                history_buffer.append(HumanMessage(content=data['contenido_mensaje']))

                        return list(reversed(history_buffer))

                    # Chains
                    ChatbotViewSet._general_llm_chain = llm | StrOutputParser()

                    ChatbotViewSet._image_chain = (
                        prompt_imagen
                        | ChatbotViewSet._general_llm_chain
                    )

                    ChatbotViewSet._video_chain = (
                        video_retriever
                        | return_video_link
                    )

                    ChatbotViewSet._history_aware_retriever_chain = (
                        condense_q_prompt 
                        | llm 
                        | StrOutputParser()
                    )

                    ChatbotViewSet._parallel_retriever_chain_manual_video = (
                        RunnableParallel(
                            video_context = lambda x: ChatbotViewSet._video_chain.invoke(x["input_query"]),
                            manual_context = lambda x: manual_retriever.invoke(x["input_query"]),
                            pregunta_usuario = lambda x: x["pregunta_usuario"],
                            image_interpretation = lambda x: x["image_interpretation"],
                            chat_history = lambda x: x["chat_history"]
                        )
                        | RunnableBranch(
                            (lambda x: x["image_interpretation"] != "", general_prompt_with_image_manual_video),
                            general_prompt_without_image_manual_video
                        )
                    )

                    ChatbotViewSet._input_chain = (
                        RunnableParallel(
                            pregunta_usuario = lambda x: x["pregunta_usuario"],
                            
                            image_interpretation = RunnableBranch(
                                (check_for_uri, ChatbotViewSet._image_chain), 
                                lambda x: ""  
                            ),
                            chat_history = RunnableLambda(fetch_history_for_chain),    
                        )
                        | RunnablePassthrough.assign(
                                                    standalone_question = RunnableBranch(
                                                        (has_history, ChatbotViewSet._history_aware_retriever_chain), 
                                                        lambda x: x["pregunta_usuario"] + " " + x["image_interpretation"]
                                                    )
                                                )
                        | RunnablePassthrough.assign(
                            input_query = lambda x: x["standalone_question"]
                        )
                    )

                    ChatbotViewSet._full_chain = (
                        ChatbotViewSet._input_chain
                        | ChatbotViewSet._parallel_retriever_chain_manual_video
                        | ChatbotViewSet._general_llm_chain
                    )

                    ChatbotViewSet._is_initialized = True
                    print("Chatbot components initialized successfully!")
                
            except Exception as e:
                error_trace = traceback.format_exc()
                print(f"Error initializing chatbot components:\n{error_trace}")

    @action(detail=False, methods=['post'])
    async def conversation(self, request):
        try:
            serializer = ChatbotMessageSerializer(data=request.data)
            
            if not ChatbotViewSet._is_initialized:
                return Response(
                    {"error": "Chatbot is still initializing. Please try again in a few moments."},
                    status=status.HTTP_503_SERVICE_UNAVAILABLE
                )

            if serializer.is_valid():
                pregunta_usuario = serializer.validated_data.get('pregunta_usuario')
                image_uri = serializer.validated_data.get('image_uri')
                idConversacion = serializer.validated_data.get('idConversacion')
                idContabilidad = serializer.validated_data.get('idContabilidad')

                if not pregunta_usuario and not image_uri:
                    return Response(
                        {
                            "status": 400,
                            "msg": "Se debe enviar una imagen o un mensaje",
                            "error": "Campos de pregunta e imagen nulos"
                        }, status=status.HTTP_400_BAD_REQUEST
                    )

                if not idConversacion or not idContabilidad:
                    return Response(
                        {
                            "status": 400,
                            "msg": "Ocurrió un error",
                            "error": "Campos de Conversacion o codigo de activación nulos"
                        }, status=status.HTTP_400_BAD_REQUEST
                    )                 

                contabilidad_obj, contabilidad_created = await Contabilidad.objects.aget_or_create(
                    id_contabilidad=idContabilidad
                )

                current_time = timezone.now()

                conv_obj, conv_created = await Conversacion.objects.aget_or_create(
                    id_conversacion=idConversacion,
                    defaults={
                        'contabilidad': contabilidad_obj,
                        'fecha_creacion': current_time,
                        'fecha_modificacion': current_time
                    }
                )

                if not conv_created:
                    conv_obj.fecha_modificacion = current_time
                    await conv_obj.asave()


                response_text = await ChatbotViewSet._full_chain.ainvoke({
                    "pregunta_usuario": pregunta_usuario,
                    "image_uri": image_uri,
                    "idConversacion": idConversacion
                })

                await Mensaje.objects.acreate(
                    conversacion=conv_obj,
                    fecha_envio=current_time,
                    contenido_mensaje=pregunta_usuario,
                    imagen_uri=image_uri, 
                    chatbot_mensaje=False
                )

                await Mensaje.objects.acreate(
                    conversacion=conv_obj,
                    fecha_envio=timezone.now(),
                    contenido_mensaje=response_text,
                    chatbot_mensaje=True 
                )

                return Response({
                        "response": response_text,
                        "status": 200,
                        "msg": "mensaje procesado con exito"
                    }, 
                    status=status.HTTP_200_OK)
        except Exception as e:
            # Log the error for debugging
            traceback.print_exc()
            return Response({"Error Answering": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @classmethod
    def cleanup(cls):
        """Cleanup method to free resources"""
        if cls._manual_vector_db or cls._video_vector_db:
            try:
                cls._manual_vector_db = None
                cls._video_vector_db = None
                # ... clean others
            except:
                pass
        cls._full_chain = None
        cls._is_initialized = False
