"""
agentes/agente_aprendizaje.py
(docstring sin cambios respecto a la versión anterior)
"""

from __future__ import annotations
from pydoc import doc
from annotated_types import doc

from pydantic import BaseModel, Field

from agent_state import AgentState, ResultadoAprendizaje
from llm_client import get_llm
from rag.vectorstore import consultar_similitud, ingerir_documento
from seguridad.rate_limiter import RateLimiter

_SYSTEM_PROMPT = """Sos un profesor especializado en temas generales.
A partir de un resumen, identificá los 3 puntos más importantes del
tema y después desarrollá una explicación pedagógica detallada
enfocada en esos 3 puntos, con tono claro y didáctico -- como si le
enseñaras el tema a alguien que no sabe nada de eso.

No agregues información que no esté en el resumen original -- tu
trabajo es EXPLICAR y estructurar, no inventar datos nuevos."""


class ExplicacionPedagogica(BaseModel):
    puntos_clave: list[str] = Field(
        description="Los 3 puntos más importantes del tema, en frases cortas"
    )
    explicacion_detallada: str = Field(
        description="Desarrollo pedagógico completo enfocado en esos 3 puntos"
    )


def _armar_contenido_embebido(explicacion: ExplicacionPedagogica) -> str:
    """Texto que se guarda/embebe en Chroma: puntos clave a modo de
    resumen arriba, desarrollo completo abajo -- mejora la densidad
    semántica para retrieval futuro por similitud."""
    puntos = "\n".join(f"- {p}" for p in explicacion.puntos_clave)
    return f"{puntos}\n\n{explicacion.explicacion_detallada}"


def construir_nodo_aprendizaje_fase1(rate_limiter: RateLimiter):
    def nodo_aprendizaje_fase1(state: AgentState) -> dict:
        tema = state["tema_consulta"]
        resultados_cache = consultar_similitud("conocimiento_aprendido", tema)

        if not resultados_cache:
            return {}

        doc = resultados_cache[0]
        # doc.page_content tiene el formato armado por _armar_contenido_embebido:
        # las líneas "- ..." al principio son los puntos clave, el resto es el
        # desarrollo. Se reconstruye separando por el primer bloque en blanco.
        bloques = doc.page_content.split("\n\n", 1)
        puntos_texto = bloques[0] if bloques else ""
        desarrollo = bloques[1] if len(bloques) > 1 else ""
        puntos_clave = [
            linea.lstrip("- ").strip()
            for linea in puntos_texto.splitlines()
            if linea.strip()
        ]

        doc = resultados_cache[0]
        resultado: ResultadoAprendizaje = {
            "puntos_clave": doc.metadata.get("puntos_clave", []),
            "explicacion_detallada": doc.page_content,
            "desde_cache": True,
            "ok": True,
        }
        
        return {"resultado_aprendizaje": resultado}

    return nodo_aprendizaje_fase1


def construir_nodo_aprendizaje_fase2(rate_limiter: RateLimiter):
    def nodo_aprendizaje_fase2(state: AgentState) -> dict:
        investigacion = state.get("resultado_investigador")

        if not investigacion or not investigacion.get("ok"):
            resultado: ResultadoAprendizaje = {
                "puntos_clave": [],
                "explicacion_detallada": "",
                "desde_cache": False,
                "ok": False,
            }
            return {"resultado_aprendizaje": resultado}

        resumen_fuente = investigacion["resumen"]
        titulo = investigacion["fuente"]

        mensajes = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": resumen_fuente},
        ]

        llm = get_llm(mensajes, rate_limiter, temperature=0.4)
        explicacion: ExplicacionPedagogica = llm.with_structured_output(
            ExplicacionPedagogica
        ).invoke(mensajes)

        contenido_embebido = _armar_contenido_embebido(explicacion)

        ingerir_documento(
            coleccion="conocimiento_aprendido",
            contenido=contenido_embebido,
            titulo_articulo=titulo,
            tool_origen="agente_aprendizaje",
            metadata_extra={
                "resumen_fuente": resumen_fuente,
                "puntos_clave": explicacion.puntos_clave,
    },
)

        resultado: ResultadoAprendizaje = {
            "puntos_clave": explicacion.puntos_clave,
            "explicacion_detallada": explicacion.explicacion_detallada,
            "desde_cache": False,
            "ok": True,
        }
        return {"resultado_aprendizaje": resultado}

    return nodo_aprendizaje_fase2