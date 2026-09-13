"""
rag/vectorstore.py

Dos colecciones de Chroma, persistidas en chroma_db/:
- wikipedia_resumenes: caché de búsquedas del Investigador
- conocimiento_aprendido: lo que el Agente de Aprendizaje enseña/guarda

Ambas comparten el mismo modelo de embeddings (rag/embeddings.py) para
que las distancias sean comparables. El umbral de similitud es un
punto de partida -- se calibra empíricamente una vez que haya datos
reales (ver spec del proyecto).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.embeddings import obtener_embeddings

DIRECTORIO_CHROMA = "./chroma_db"

NombreColeccion = Literal["wikipedia_resumenes", "conocimiento_aprendido"]

UMBRAL_SIMILITUD = 0.75

_vectorstores: dict[str, Chroma] = {}


def _obtener_vectorstore(coleccion: NombreColeccion) -> Chroma:
    """Devuelve (o crea) la instancia de Chroma para esa colección puntual."""
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
    """
    Busca en la colección indicada los documentos más parecidos a
    tema_consulta. Devuelve solo los que superan `umbral`.
    """
    vectorstore = _obtener_vectorstore(coleccion)
    resultados = vectorstore.similarity_search_with_relevance_scores(tema_consulta, k=k)
    return [doc for doc, score in resultados if score >= umbral]


def ingerir_documento(
    coleccion: NombreColeccion,
    contenido: str,
    titulo_articulo: str,
    tool_origen: str,
    metadata_extra: dict | None = None,
    id_documento: str | None = None,
) -> None:
    """
    Embebe y guarda un documento nuevo en la colección indicada.

    metadata mínima (definida en el spec):
      - titulo_articulo: el título usado para llegar a este contenido
      - fecha_ingesta: cuándo se guardó (ISO 8601, UTC)
      - tool_origen: qué tool/agente lo generó (ej. "obtener_resumen",
        "agente_aprendizaje")

    metadata_extra: campos adicionales opcionales, específicos de quién
    llama (ej. agente_aprendizaje.py guarda ahí resumen_fuente y
    puntos_clave, separado del contenido pedagógico que se embebe).

    NOTA: no incluye fuente_url -- servidor_wiki.py no la provee (las
    tools devuelven texto plano, no URLs de artículo).
    
    (docstring igual, agregar:)
    id_documento: opcional. Si se pasa, Chroma sobrescribe el documento
    existente con ese id en vez de crear uno nuevo -- útil para pruebas
    manuales repetidas (evita duplicados en cada corrida).

    """
    vectorstore = _obtener_vectorstore(coleccion)
    metadata = {
        "titulo_articulo": titulo_articulo,
        "fecha_ingesta": datetime.now(timezone.utc).isoformat(),
        "tool_origen": tool_origen,
    }
    if metadata_extra:
        metadata.update(metadata_extra)

    documento = Document(page_content=contenido, metadata=metadata)

    if id_documento:
        vectorstore.add_documents([documento], ids=[id_documento])
    else:
        vectorstore.add_documents([documento])


if __name__ == "__main__":
    print("Ingresando documento de prueba en 'wikipedia_resumenes'...")
    ingerir_documento(
         coleccion="wikipedia_resumenes",
         contenido="Un agujero negro es un objeto con un campo gravitacional tan fuerte que ni la luz puede escapar.",
         titulo_articulo="Agujero negro",
         tool_origen="obtener_resumen",
         id_documento="test-agujero-negro",
    )

    print("Consultando por un tema similar...")
    resultados = consultar_similitud("wikipedia_resumenes", "qué es un agujero negro")
    for doc in resultados:
        print(f" - {doc.metadata['titulo_articulo']}: {doc.page_content[:80]}...")

    if not resultados:
        print(" (sin resultados sobre el umbral -- revisar UMBRAL_SIMILITUD)")

    # --- Prueba nueva: metadata_extra con lista (puntos_clave) ---
    print("\nProbando metadata_extra con una lista...")
    ingerir_documento(
        coleccion="conocimiento_aprendido",
        contenido="Prueba con puntos clave",
        titulo_articulo="Test",
        tool_origen="test_manual",
        metadata_extra={"puntos_clave": ["uno", "dos", "tres"]},
        id_documento="test-puntos-clave",
    )
    resultados_test = consultar_similitud("conocimiento_aprendido", "prueba", umbral=0.0)
    for doc in resultados_test:
        print(f" - puntos_clave: {doc.metadata.get('puntos_clave')} (tipo: {type(doc.metadata.get('puntos_clave'))})")