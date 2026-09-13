"""
Cliente_agemte.py

Cliente MCP reusable: conexión a servidor_wiki.py + carga de tools
para LangChain + interpretación de resultados. Los 3 agentes importan
estas funciones en vez de reimplementar la conexión cada uno.
"""

import asyncio
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools


@asynccontextmanager
async def conectar_mcp():
    """
    Conecta con el servidor MCP de Wikipedia (servidor_wiki.py) vía stdio
    y entrega una ClientSession ya inicializada, lista para usar.
    """
    parametros_servidor = StdioServerParameters(
        command="python",
        args=["servidor_wiki.py"],
    )

    async with stdio_client(parametros_servidor) as (lectura, escritura):
        async with ClientSession(lectura, escritura) as sesion:
            await sesion.initialize()
            yield sesion


async def obtener_tools_langchain(sesion: ClientSession):
    """
    Carga las tools expuestas por servidor_wiki.py ya envueltas como
    BaseTool de LangChain, listas para bind_tools().
    """
    return await load_mcp_tools(sesion)


# --- NUEVA: acá va la función que agregamos ---
def extraer_texto_resultado(resultado_tool) -> str:
    """load_mcp_tools devuelve el contenido como lista de dicts
    [{'type': 'text', 'text': '...'}] -- extrae el texto plano."""
    if isinstance(resultado_tool, list) and resultado_tool:
        primero = resultado_tool[0]
        if isinstance(primero, dict) and "text" in primero:
            return primero["text"]
    return str(resultado_tool)
# --- fin de la función nueva ---


async def _prueba_conexion():
    """
    Prueba manual de conexión y listado de tools. No se usa en
    producción -- sirve para verificar que el servidor responde.
    """
    print("Iniciando conexión con el Servidor MCP...")

    async with conectar_mcp() as sesion:
        print("✅ Conectado exitosamente.\n")

        tools = await obtener_tools_langchain(sesion)

        print("🛠️ Herramientas cargadas para LangChain:")
        for tool in tools:
            print(f" - Nombre: {tool.name}")
            print(f" - Descripción: {tool.description}\n")

        tema_a_buscar = "Agujero negro"
        tool_busqueda = next(t for t in tools if t.name == "buscar_articulos")
        print(f"🤖 Ejecutando '{tool_busqueda.name}' para el tema: '{tema_a_buscar}'...\n")

        resultado = await tool_busqueda.ainvoke({"consulta": tema_a_buscar})

        print("📄 Datos devueltos:")
        print(extraer_texto_resultado(resultado))


if __name__ == "__main__":
    asyncio.run(_prueba_conexion())