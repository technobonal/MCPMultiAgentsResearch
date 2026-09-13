"""
agent_state.py

Estado compartido del grafo (StateGraph de LangGraph). Cada nodo del
sistema (validación, supervisor, los 3 agentes, respuesta final) lee y
escribe sobre esta misma estructura.
"""

from __future__ import annotations

from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages


class ResultadoInvestigador(TypedDict, total=False):
    resumen: str
    articulos_relacionados: list[str]
    fuente: str
    desde_cache: bool
    ok: bool


class ResultadoMultimedia(TypedDict, total=False):
    imagenes: list[str]
    ok: bool


class ResultadoAprendizaje(TypedDict, total=False):
    puntos_clave: list[str]          # los 3 puntos más importantes del tema
    explicacion_detallada: str        # desarrollo pedagógico completo
    desde_cache: bool
    ok: bool


class PasoInvestigacion(TypedDict):
    intento: int
    query: str
    resultado_ok: bool


class AgentState(TypedDict):
    mensajes: Annotated[list, add_messages]
    thread_id: str

    input_sanitizado: Optional[str]

    tema_consulta: Optional[str]
    tipo_solicitud: Optional[str]  # "informacion_general" | "solo_imagenes"

    resultado_investigador: Optional[ResultadoInvestigador]
    resultado_multimedia: Optional[ResultadoMultimedia]
    resultado_aprendizaje: Optional[ResultadoAprendizaje]

    pasos_investigacion: Annotated[list[PasoInvestigacion], add_messages]

    fuentes: Optional[list[str]]
    respuesta_final: Optional[str]

    estado: str
    error: Optional[str]