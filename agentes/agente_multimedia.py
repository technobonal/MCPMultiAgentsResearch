"""
agentes/agente_multimedia.py

Nodo del fan-out encargado de conseguir imágenes de tema_consulta.

Más simple que el Investigador: una sola tool (obtener_imagenes), sin
loop ReAct multi-turno -- alcanza con una llamada del LLM que decida
invocar la tool con el tema, y parsear el texto plano que devuelve en
una lista de URLs.

Escribe en resultado_multimedia, según el contrato de agent_state.py.
"""

from __future__ import annotations

from agent_state import AgentState, ResultadoMultimedia
from Cliente_agemte import conectar_mcp, obtener_tools_langchain, extraer_texto_resultado, con_reintento_mcp
from llm_client import get_llm
from seguridad.rate_limiter import RateLimiter

_SYSTEM_PROMPT = """Sos el agente Multimedia de un sistema sobre Wikipedia.
Tenés una sola herramienta: obtener_imagenes, que recibe el título
exacto de un artículo y devuelve hasta 3 URLs de imágenes.

Llamá a obtener_imagenes con el tema que te dan como título."""


def _parsear_urls(texto_tool: str) -> list[str]:
    """
    obtener_imagenes devuelve texto plano tipo:
        "Imágenes encontradas:\n- https://...\n- https://...\n"
    o mensajes de error/sin resultados sin URLs. Extrae las líneas que
    arrancan con "- http" y devuelve solo la URL.
    """
    urls = []
    for linea in texto_tool.splitlines():
        linea = linea.strip()
        if linea.startswith("- http"):
            urls.append(linea[2:].strip())
    return urls


def construir_nodo_multimedia(rate_limiter: RateLimiter):
    """Factory -- mismo patrón que construir_nodo_supervisor y
    construir_nodo_investigador, recibe el RateLimiter compartido."""

    async def nodo_multimedia(state: AgentState) -> dict:
        tema = state["tema_consulta"]

        async def _intentar():
            async with conectar_mcp() as sesion:
                tools = await obtener_tools_langchain(sesion)
                tools_multimedia = [t for t in tools if t.name == "obtener_imagenes"]

                mensajes = [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": tema},
                ]

                llm = get_llm(mensajes, rate_limiter)
                llm_con_tools = llm.bind_tools(tools_multimedia)

                respuesta = await llm_con_tools.ainvoke(mensajes)

                texto_resultado = ""
                if respuesta.tool_calls:
                    tool_call = respuesta.tool_calls[0]
                    tool = tools_multimedia[0]
                    resultado_crudo = await tool.ainvoke(tool_call["args"])
                    texto_resultado = extraer_texto_resultado(resultado_crudo)

            urls = _parsear_urls(texto_resultado)
            resultado: ResultadoMultimedia = {
                "imagenes": urls,
                "ok": len(urls) > 0,
            }
            return {"resultado_multimedia": resultado}

        try:
            return await con_reintento_mcp(_intentar)
        except Exception:
            resultado: ResultadoMultimedia = {"imagenes": [], "ok": False}
            return {"resultado_multimedia": resultado}

    return nodo_multimedia