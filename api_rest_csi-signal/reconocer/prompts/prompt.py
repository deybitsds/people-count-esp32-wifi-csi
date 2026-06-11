from langchain_core.prompts import ChatPromptTemplate

template_imagen = '''
Reconoce los campos de la boleta/factura
1. Devolver "none" si no hay información del campo. No inventes nada
2. Subtotal solo existe si OP. Gravadas | Op. Exoneradas y OP. Inafectadas no existen
3. La moneda en texto: 
    - "sol": S/. | S/ | no haya moneda (valor predeterminado)
    - "dolar": $
    - "euro": €
4. Solo devuelve el formato, sin texto
5. Formato exacto:
{{
  "nro_factura": "...",
  "ruc_proveedor": "...",
  "ruc_cliente": "...",
  "op_gravadas": "...",
  "op_exoneradas": "...",
  "op_inafectadas": "...",
  "monto_subtotal": "...",
  "monto_total": "...",
  "igv": "...",
  "fecha_emision": "...",
  "fecha_vencimiento": "...",
  "moneda": "..."
}}
'''

prompt_imagen = ChatPromptTemplate.from_messages([
    (
        "user", 
        [
            {"type": "text", "text": template_imagen},
            {"type": "image_url", "image_url": {"url": "{img_bin64}"}}
        ]
    )
])

# =========================================================
# ====================         2       ====================
#
