"""
agent_state.py
"""

from __future__ import annotations

import operator
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
    puntos_clave: list[str]
    explicacion_detallada: str
    desde_cache: bool
    ok: bool


class PasoInvestigacion(TypedDict):
    intento: int
    query: str
    resultado_ok: bool


class AgentState(TypedDict):
    mensajes: Annotated[list, add_messages]  # este SÍ son mensajes de chat reales -- queda igual
    thread_id: str

    input_sanitizado: Optional[str]

    tema_consulta: Optional[str]
    tipo_solicitud: Optional[str]

    resultado_investigador: Optional[ResultadoInvestigador]
    resultado_multimedia: Optional[ResultadoMultimedia]
    resultado_aprendizaje: Optional[ResultadoAprendizaje]

    # operator.add en vez de add_messages: PasoInvestigacion NO es un
    # mensaje de chat, es un dict propio -- solo necesitamos que las
    # listas de cada superstep se concatenen, no que se coercionen a
    # BaseMessage.
    pasos_investigacion: Annotated[list[PasoInvestigacion], operator.add]

    fuentes: Optional[list[str]]
    respuesta_final: Optional[str]

    estado: str
    error: Optional[str]