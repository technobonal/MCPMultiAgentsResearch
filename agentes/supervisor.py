"""
agentes/supervisor.py

Nodo orquestador del grafo. Responsabilidades:
1. Clasifica la intención del usuario UNA SOLA VEZ, vía structured
   output (IntencionOut) -- evita que cada agente tenga que
   re-derivar el tema por su cuenta.
2. Deja tema_consulta en el AgentState para que el fan-out lo lea
   directo, sin que el supervisor tenga que "pasárselo" a cada agente
   a mano (eso lo resuelve grafo.py con los edges).

El supervisor NO ejecuta tools ni llama a Wikipedia -- solo clasifica
y decide el routing. La ejecución real vive en cada agente.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agent_state import AgentState
from llm_client import get_llm
from seguridad.rate_limiter import RateLimiter

_SYSTEM_PROMPT = """Sos el supervisor de un sistema de investigación sobre Wikipedia.
Tu único trabajo es leer el pedido del usuario y clasificarlo en una
estructura, NO respondés directamente al usuario.

Extraé:
- tema_consulta: el tema o entidad concreta que hay que investigar,
  normalizado (sin muletillas como "contame sobre" o "quiero saber de").
- tipo_solicitud:
    "informacion_general" -> el usuario quiere texto/explicación sobre el tema
    "solo_imagenes" -> el usuario pide específicamente imágenes/fotos del tema
- urgencia: opcional, solo si el usuario la expresa explícitamente
  (ej. "necesito esto ya"); en general dejarlo vacío.

Si el pedido no tiene un tema claro, hacé tu mejor inferencia del tema
más probable a partir del texto."""


class IntencionOut(BaseModel):
    tema_consulta: str = Field(description="Tema normalizado a investigar")
    tipo_solicitud: Literal["informacion_general", "solo_imagenes"] = Field(
        description="Qué tipo de resultado espera el usuario"
    )
    urgencia: str | None = Field(
        default=None, description="Urgencia expresada explícitamente por el usuario, si la hay"
    )


def construir_nodo_supervisor(rate_limiter: RateLimiter):
    """
    Factory: recibe la instancia ÚNICA y compartida de RateLimiter (la
    misma que van a usar los 3 agentes) y devuelve la función de nodo
    lista para agregar al StateGraph con builder.add_node("supervisor", ...).
    """

    def nodo_supervisor(state: AgentState) -> dict:
        texto_usuario = state["input_sanitizado"]

        mensajes = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": texto_usuario},
        ]

        llm = get_llm(mensajes, rate_limiter, temperature=0.0)
        clasificador = llm.with_structured_output(IntencionOut)

        intencion: IntencionOut = clasificador.invoke(mensajes)

        return {
            "tema_consulta": intencion.tema_consulta,
            "tipo_solicitud": intencion.tipo_solicitud,
            "estado": "enrutando",
        }

    return nodo_supervisor