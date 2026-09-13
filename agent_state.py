"""
agent_state.py

Estado compartido del grafo (StateGraph de LangGraph). Cada nodo del
sistema (validación, supervisor, los 3 agentes, respuesta final) lee y
escribe sobre esta misma estructura.

Puntos clave del diseño (ver spec del proyecto):
- Cada agente del fan-out escribe en SU PROPIA clave (resultado_*),
  nunca en una compartida -- evita que se pisen entre sí al correr en
  paralelo.
- pasos_investigacion usa el reducer add_messages para ACUMULAR cada
  intento del loop ReAct del Investigador, no sobrescribir -- así
  queda trazable el ir-y-volver completo, no solo el último intento.
- mensajes también usa add_messages: es el historial del hilo,
  persistido por el checkpointer (SqliteSaver) entre turnos.
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages


class ResultadoInvestigador(TypedDict, total=False):
    resumen: str
    articulos_relacionados: list[str]
    fuente: str  # título del artículo usado, no una URL (la tool no la provee)
    desde_cache: bool
    ok: bool


class ResultadoMultimedia(TypedDict, total=False):
    imagenes: list[str]  # URLs
    ok: bool


class ResultadoAprendizaje(TypedDict, total=False):
    puntos_clave: list[str]  # los 3 puntos más importantes, frases cortas
    explicacion_detallada: str  # desarrollo pedagógico completo, tono de profesor
    desde_cache: bool
    ok: bool


class PasoInvestigacion(TypedDict):
    intento: int
    query: str
    resultado_ok: bool


class AgentState(TypedDict):
    # --- Hilo de conversación (memoria corta, persistida por el checkpointer) ---
    mensajes: Annotated[list, add_messages]
    thread_id: str

    # --- Validación de input (primer nodo del grafo) ---
    input_sanitizado: Optional[str]

    # --- Clasificación del supervisor (IntencionOut) ---
    tema_consulta: Optional[str]
    tipo_solicitud: Optional[str]  # "informacion_general" | "solo_imagenes" | "sorpresa"

    # --- Resultados del fan-out (una clave por agente, sin choques) ---
    resultado_investigador: Optional[ResultadoInvestigador]
    resultado_multimedia: Optional[ResultadoMultimedia]
    resultado_aprendizaje: Optional[ResultadoAprendizaje]

    # --- Trazabilidad del loop interno del Investigador (se acumula, no se sobrescribe) ---
    pasos_investigacion: Annotated[list[PasoInvestigacion], operator.add]

    # --- Salida final ---
    fuentes: Optional[list[str]]
    respuesta_final: Optional[str]

    # --- Control de estado de la solicitud ---
    estado: str  # recibido | rechazado | enrutando | ejecutando | completado | error
    error: Optional[str]