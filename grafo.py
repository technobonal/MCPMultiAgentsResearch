"""
grafo.py

Corrección de sincronización: LangGraph solo fusiona ramas que llegan
al mismo nodo en el MISMO superstep -- si llegan en supersteps
distintos, el nodo destino se ejecuta una vez por cada llegada (bug
confirmado, ver forum.langchain.com y langgraphjs#1535).

Como investigador->aprendizaje_fase2 tiene un hop más que multimedia,
agregamos un nodo passthrough "esperar_multimedia" para igualar la
profundidad de las dos ramas antes de converger en respuesta_final.

aprendizaje_fase1 ya NO tiene ruta directa a respuesta_final: solo
escribe resultado_aprendizaje en el estado compartido si hay match en
Chroma. aprendizaje_fase2 lee ese estado y, si ya está resuelto
(cache hit), no hace su propio trabajo -- solo entonces pasa a
respuesta_final, manteniendo un ÚNICO punto de convergencia real.
"""

from __future__ import annotations

import os
import tempfile

import aiosqlite

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph


from agent_state import AgentState
from agentes.agente_aprendizaje import (
    construir_nodo_aprendizaje_fase1,
    construir_nodo_aprendizaje_fase2,
)
from agentes.agente_investigador import construir_nodo_investigador
from agentes.agente_multimedia import construir_nodo_multimedia
from agentes.supervisor import construir_nodo_supervisor
from seguridad.rate_limiter import RateLimiter
from seguridad.validadores import MENSAJE_RECHAZO, procesar_input_usuario


def nodo_validacion(state: AgentState) -> dict:
    print("[DEBUG] entrando a nodo_validacion", flush=True)
    texto_crudo = state["mensajes"][-1].content
    input_sanitizado, rechazado = procesar_input_usuario(texto_crudo)
    print(f"[DEBUG] saliendo de nodo_validacion, rechazado={rechazado}", flush=True)

    if rechazado:
        return {"estado": "rechazado", "respuesta_final": MENSAJE_RECHAZO}
    return {"input_sanitizado": input_sanitizado, "estado": "enrutando"}


def enrutar_tras_validacion(state: AgentState) -> str:
    return "rechazado" if state["estado"] == "rechazado" else "ok"


def enrutar_tras_supervisor(state: AgentState) -> list[str]:
    tipo = state["tipo_solicitud"]
    if tipo == "solo_imagenes":
        return ["multimedia_padded"]
    if tipo == "explicacion_sin_imagen":
        return ["investigador", "aprendizaje_fase1"]
    return ["investigador", "aprendizaje_fase1", "multimedia_padded"]


def nodo_esperar_multimedia(state: AgentState) -> dict:
    """
    Passthrough: no hace nada nuevo, solo iguala la profundidad de la
    rama de Multimedia con la de investigador->aprendizaje_fase2 (2
    hops cada una), para que ambas lleguen a respuesta_final en el
    mismo superstep -- sin esto, respuesta_final se dispara 2 veces.
    """
    return {}


def nodo_respuesta_final(state: AgentState) -> dict:
    partes = []
    fuentes = []

    aprendizaje = state.get("resultado_aprendizaje")
    investigacion = state.get("resultado_investigador")

    if aprendizaje and aprendizaje.get("ok"):
        puntos_unicos = list(dict.fromkeys(aprendizaje.get("puntos_clave", [])))  # dedup preservando orden
        puntos = "\n".join(f"- {p}" for p in puntos_unicos)
        partes.append(f"## Puntos clave\n\n{puntos}\n\n## Explicación\n\n{aprendizaje['explicacion_detallada']}")
    elif investigacion and investigacion.get("ok"):
        partes.append(investigacion["resumen"])

    if investigacion and investigacion.get("fuente"):
        fuentes.append(investigacion["fuente"])

    multimedia = state.get("resultado_multimedia")
    if not partes and multimedia and multimedia.get("ok"):
        partes.append("Acá tenés las imágenes que encontré.")

    return {
        "respuesta_final": "\n\n".join(partes) if partes else "No se encontró información.",
        "fuentes": fuentes,
        "estado": "completado",
    }


async def construir_grafo(rate_limiter: RateLimiter):
    builder = StateGraph(AgentState)

    builder.add_node("validacion", nodo_validacion)
    builder.add_node("supervisor", construir_nodo_supervisor(rate_limiter))
    builder.add_node("investigador", construir_nodo_investigador(rate_limiter))
    builder.add_node("multimedia_padded", construir_nodo_multimedia(rate_limiter))
    builder.add_node("esperar_multimedia", nodo_esperar_multimedia)
    builder.add_node("aprendizaje_fase1", construir_nodo_aprendizaje_fase1(rate_limiter))
    builder.add_node("aprendizaje_fase2", construir_nodo_aprendizaje_fase2(rate_limiter))
    builder.add_node("respuesta_final", nodo_respuesta_final)

    builder.set_entry_point("validacion")

    builder.add_conditional_edges(
        "validacion", enrutar_tras_validacion, {"rechazado": END, "ok": "supervisor"},
    )
    builder.add_conditional_edges(
        "supervisor", enrutar_tras_supervisor,
        ["investigador", "aprendizaje_fase1", "multimedia_padded"],
    )
    builder.add_edge("investigador", "aprendizaje_fase2")
    builder.add_edge("aprendizaje_fase2", "respuesta_final")
    builder.add_edge("multimedia_padded", "esperar_multimedia")
    builder.add_edge("esperar_multimedia", "respuesta_final")
    builder.add_edge("respuesta_final", END)

    ruta_checkpoints = os.path.join(tempfile.gettempdir(), "mcpma_checkpoints.sqlite")
    conexion = await aiosqlite.connect(ruta_checkpoints)
    checkpointer = AsyncSqliteSaver(conexion)

    return builder.compile(checkpointer=checkpointer)