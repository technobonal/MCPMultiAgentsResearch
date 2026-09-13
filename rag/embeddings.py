"""
rag/embeddings.py

Modelo de embeddings local (HuggingFace, sin API externa más allá de
la descarga inicial desde huggingface.co -- después corre 100% local).

Se carga como singleton perezoso (_modelo global) porque instanciar
HuggingFaceEmbeddings implica cargar el modelo en memoria -- no
conviene hacerlo más de una vez por proceso.
"""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

MODELO_EMBEDDINGS = "intfloat/multilingual-e5-base"

_modelo: HuggingFaceEmbeddings | None = None


def obtener_embeddings() -> HuggingFaceEmbeddings:
    """
    Devuelve la instancia única del modelo de embeddings, cargándolo
    la primera vez que se pide (después queda en memoria para el resto
    del proceso). Usado tanto por rag/vectorstore.py para ingesta como
    para consulta -- mismo modelo en los dos casos, obligatorio para
    que las distancias sean comparables.
    """
    global _modelo
    if _modelo is None:
        _modelo = HuggingFaceEmbeddings(
            model_name=MODELO_EMBEDDINGS,
            encode_kwargs={"normalize_embeddings": True},  # necesario para que el score de similitud sea coseno 0-1
        )
    return _modelo