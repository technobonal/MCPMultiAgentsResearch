"""
rag/vectorstore.py
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.embeddings import obtener_embeddings

DIRECTORIO_CHROMA = "./chroma_db"
NombreColeccion = Literal["wikipedia_resumenes", "conocimiento_aprendido"]
UMBRAL_SIMILITUD = 0.75

_vectorstores: dict[str, Chroma] = {}


def _obtener_vectorstore(coleccion: NombreColeccion) -> Chroma:
    if coleccion not in _vectorstores:
        _vectorstores[coleccion] = Chroma(
            collection_name=coleccion,
            embedding_function=obtener_embeddings(),
            persist_directory=DIRECTORIO_CHROMA,
        )
    return _vectorstores[coleccion]


def consultar_similitud(
    coleccion: NombreColeccion,
    tema_consulta: str,
    k: int = 4,
    umbral: float = UMBRAL_SIMILITUD,
) -> list[Document]:
    vectorstore = _obtener_vectorstore(coleccion)
    resultados = vectorstore.similarity_search_with_relevance_scores(tema_consulta, k=k)
    return [doc for doc, score in resultados if score >= umbral]


def ingerir_documento(
    coleccion: NombreColeccion,
    contenido: str,
    titulo_articulo: str,
    tool_origen: str,
    metadata_extra: dict[str, Any] | None = None,
) -> None:
    """
    metadata_extra: campos adicionales específicos de quien ingesta
    (ej. agente_aprendizaje.py guarda ahí puntos_clave y
    resumen_fuente, para poder reconstruir ResultadoAprendizaje
    completo en un cache hit sin tener que re-parsear page_content).
    """
    metadata = {
        "titulo_articulo": titulo_articulo,
        "fecha_ingesta": datetime.now(timezone.utc).isoformat(),
        "tool_origen": tool_origen,
    }
    if metadata_extra:
        metadata.update(metadata_extra)

    vectorstore = _obtener_vectorstore(coleccion)
    documento = Document(page_content=contenido, metadata=metadata)
    vectorstore.add_documents([documento])