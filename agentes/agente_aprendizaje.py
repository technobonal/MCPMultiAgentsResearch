"""
agentes/agente_aprendizaje.py

Fase 1 (paralela, sin ruta directa a respuesta_final): consulta Chroma
por tema_consulta en la colección "conocimiento_aprendido". Si hay
match, escribe resultado_aprendizaje directo desde el estado
compartido -- no hace falta llamar al LLM.

Fase 2 (después de investigador): si Fase 1 no encontró nada, toma
resultado_investigador, genera una explicación pedagógica (rol de
"profesor") y la guarda en Chroma para que futuras consultas similares
peguen en Fase 1.
"""

from __future__ import annotations

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
    """Texto que se embebe en Chroma: puntos clave arriba, desarrollo
    completo abajo -- mejora la densidad semántica para retrieval futuro."""
    puntos = "\n".join(f"- {p}" for p in explicacion.puntos_clave)
    return f"{puntos}\n\n{explicacion.explicacion_detallada}"


def construir_nodo_aprendizaje_fase1(rate_limiter: RateLimiter):
    def nodo_aprendizaje_fase1(state: AgentState) -> dict:
        tema = state["tema_consulta"]
        resultados_cache = consultar_similitud("conocimiento_aprendido", tema)

        if not resultados_cache:
            return {}

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
        # Si Fase 1 ya resolvió (cache hit), no hay nada que hacer acá.
        if state.get("resultado_aprendizaje") is not None:
            return {}

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
            metadata_extra={"puntos_clave": explicacion.puntos_clave},
        )

        resultado: ResultadoAprendizaje = {
            "puntos_clave": explicacion.puntos_clave,
            "explicacion_detallada": explicacion.explicacion_detallada,
            "desde_cache": False,
            "ok": True,
        }
        return {"resultado_aprendizaje": resultado}

    return nodo_aprendizaje_fase2