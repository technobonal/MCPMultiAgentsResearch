"""
agentes/agente_investigador.py

Nodo del fan-out encargado de investigar tema_consulta, vía un loop
ReAct acotado con las tools buscar_articulos/obtener_resumen.

Este agente NO toca Chroma -- ni consulta caché ni ingesta. Esa
responsabilidad es exclusiva de agente_aprendizaje.py (Fase 1 y Fase
2), para que ambos agentes puedan mostrarse trabajando en paralelo sin
pisarse ni duplicar lógica.

MAX_INTENTOS = 3 -> 1 intento inicial + 2 reintentos, según lo
decidido para arista_verificacion.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent_state import AgentState, PasoInvestigacion, ResultadoInvestigador
from Cliente_agemte import conectar_mcp, obtener_tools_langchain, extraer_texto_resultado, con_reintento_mcp
from llm_client import get_llm
from seguridad.rate_limiter import RateLimiter

TOOLS_PERMITIDAS = {"buscar_articulos", "obtener_resumen"}
MAX_INTENTOS = 3

_SYSTEM_PROMPT = """Sos el agente Investigador de un sistema sobre Wikipedia.
Tenés dos herramientas: buscar_articulos (para encontrar el título
exacto de un artículo a partir de una consulta) y obtener_resumen
(para traer el texto del artículo, dado un título exacto).

Tu objetivo es conseguir un resumen útil sobre el tema pedido. Si no
conocés el título exacto, usá primero buscar_articulos; una vez que
tengas un título, llamá a obtener_resumen con ese título."""


def construir_nodo_investigador(rate_limiter: RateLimiter):
    """Factory: recibe el RateLimiter compartido, devuelve la función de nodo async."""

    async def nodo_investigador(state: AgentState) -> dict:
        tema = state["tema_consulta"]

        async def _intentar():
            async with conectar_mcp() as sesion:
                todas_las_tools = await obtener_tools_langchain(sesion)
                tools_investigador = [t for t in todas_las_tools if t.name in TOOLS_PERMITIDAS]

                mensajes: list = [
                    SystemMessage(content=_SYSTEM_PROMPT),
                    HumanMessage(content=tema),
                ]

                llm = get_llm(
                    [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": tema}],
                    rate_limiter,
                )
                llm_con_tools = llm.bind_tools(tools_investigador)

                pasos: list[PasoInvestigacion] = []
                resumen_final: str | None = None
                fuente_final: str | None = None

                for intento in range(1, MAX_INTENTOS + 1):
                    respuesta: AIMessage = await llm_con_tools.ainvoke(mensajes)
                    mensajes.append(respuesta)

                    if not respuesta.tool_calls:
                        resumen_final = respuesta.content or None
                        pasos.append({"intento": intento, "query": tema, "resultado_ok": bool(resumen_final)})
                        break

                    for tool_call in respuesta.tool_calls:
                        tool = next(t for t in tools_investigador if t.name == tool_call["name"])
                        resultado_crudo = await tool.ainvoke(tool_call["args"])
                        texto_resultado = extraer_texto_resultado(resultado_crudo)

                        mensajes.append(ToolMessage(content=texto_resultado, tool_call_id=tool_call["id"]))

                        if tool_call["name"] == "obtener_resumen" and "Error" not in texto_resultado:
                            resumen_final = texto_resultado
                            fuente_final = tool_call["args"].get("titulo")

                    pasos.append({"intento": intento, "query": tema, "resultado_ok": resumen_final is not None})
                    if resumen_final:
                        break

            resultado: ResultadoInvestigador = {
                "resumen": resumen_final or "No se encontró información suficiente.",
                "articulos_relacionados": [],
                "fuente": fuente_final or tema,
                "desde_cache": False,
                "ok": resumen_final is not None,
            }
            return {"resultado_investigador": resultado, "pasos_investigacion": pasos}

        try:
            return await con_reintento_mcp(_intentar)
        except Exception:
            resultado: ResultadoInvestigador = {
                "resumen": "No se pudo conectar con el servidor de Wikipedia.",
                "articulos_relacionados": [],
                "fuente": tema,
                "desde_cache": False,
                "ok": False,
            }
            return {"resultado_investigador": resultado, "pasos_investigacion": []}

    return nodo_investigador


if __name__ == "__main__":
    import asyncio

    async def _prueba():
        rate_limiter = RateLimiter()
        nodo = construir_nodo_investigador(rate_limiter)

        estado_falso = {
            "tema_consulta": "agujero negro",
            "mensajes": [], "thread_id": "test", "input_sanitizado": None,
            "tipo_solicitud": None, "resultado_investigador": None,
            "resultado_multimedia": None, "resultado_aprendizaje": None,
            "pasos_investigacion": [], "fuentes": None, "respuesta_final": None,
            "estado": "enrutando", "error": None,
        }

        resultado = await nodo(estado_falso)
        print("resultado_investigador:", resultado["resultado_investigador"])
        print("pasos_investigacion:", resultado["pasos_investigacion"])

    asyncio.run(_prueba())