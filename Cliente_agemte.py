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

    Uso desde cualquier agente:
        async with conectar_mcp() as sesion:
            tools = await obtener_tools_langchain(sesion)
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
    BaseTool de LangChain (buscar_articulos, obtener_resumen,
    obtener_imagenes, aprender_algo_nuevo), listas para bind_tools().

    Cada agente decide cuáles usar filtrando esta lista por nombre
    (aislamiento por capacidades: cada agente solo se queda con las
    tools que le corresponden).
    """
    return await load_mcp_tools(sesion)


async def _prueba_conexion():
    """
    Prueba manual de conexión y listado de tools. No se usa en
    producción — sirve para verificar que el servidor responde y que
    langchain-mcp-adapters carga las tools correctamente.
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
        print(resultado)


if __name__ == "__main__":
    asyncio.run(_prueba_conexion())