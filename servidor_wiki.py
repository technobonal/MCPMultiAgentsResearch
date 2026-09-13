import asyncio
import httpx
from mcp.server.fastmcp import FastMCP as MCPServer

mcp = MCPServer("AgenteWikipedia")

WIKI_REST_URL = "https://es.wikipedia.org/api/rest_v1"
HEADERS = {
    "User-Agent": "AgenteWikipedia_MCP_Alura/1.0 (contacto@tuemail.com)"
}

@mcp.tool()
async def buscar_articulos(consulta: str) -> str:
    """Busca artículos en Wikipedia usando una frase o palabra clave."""
    search_url = f"https://es.wikipedia.org/w/rest.php/v1/search/page?q={consulta}&limit=3"
    async with httpx.AsyncClient(follow_redirects=True) as client:
        respuesta = await client.get(search_url, headers=HEADERS)
        if respuesta.status_code == 200:
            datos = respuesta.json()
            paginas = datos.get("pages", [])
            if not paginas:
                return "No se encontraron artículos para esa búsqueda."
            resultado = "Artículos encontrados:\n"
            for p in paginas:
                resultado += f"- {p['title']}: {p['description']}\n"
            return resultado
        return f"Error al buscar: {respuesta.status_code}"

@mcp.tool()
async def obtener_resumen(titulo: str) -> str:
    """Obtiene el resumen principal de un artículo."""
    url = f"{WIKI_REST_URL}/page/summary/{titulo}"
    async with httpx.AsyncClient(follow_redirects=True) as client:
        respuesta = await client.get(url, headers=HEADERS)
        if respuesta.status_code == 200:
            datos = respuesta.json()
            return datos.get("extract", "No hay resumen disponible.")
        return f"Error: No se pudo encontrar el artículo '{titulo}'."

@mcp.tool()
async def obtener_imagenes(titulo: str) -> str:
    """Obtiene los enlaces a las imágenes principales de un artículo."""
    # ¡Ruta corregida a media-list!
    url = f"{WIKI_REST_URL}/page/media-list/{titulo}"
    async with httpx.AsyncClient(follow_redirects=True) as client:
        respuesta = await client.get(url, headers=HEADERS)
        if respuesta.status_code == 200:
            datos = respuesta.json()
            items = datos.get("items", [])
            if not items:
                return "No se encontraron imágenes en este artículo."
            
            resultado = "Imágenes encontradas:\n"
            contador = 0
            for item in items: 
                # Buscamos específicamente los archivos que sean imágenes
                if item.get("type") == "image" and "srcset" in item:
                    url_imagen = item["srcset"][0]["src"]
                    # Wikipedia a veces devuelve la URL sin el https:, se lo agregamos
                    if url_imagen.startswith("//"):
                        url_imagen = "https:" + url_imagen
                    resultado += f"- {url_imagen}\n"
                    contador += 1
                    if contador == 3: # Máximo 3 imágenes
                        break
            return resultado
        return f"Error al buscar imágenes. Código: {respuesta.status_code}."

@mcp.tool()
async def aprender_algo_nuevo() -> str:
    """Obtiene un artículo completamente al azar."""
    url = f"{WIKI_REST_URL}/page/random/summary"
    async with httpx.AsyncClient(follow_redirects=True) as client:
        respuesta = await client.get(url, headers=HEADERS)
        if respuesta.status_code == 200:
            datos = respuesta.json()
            titulo = datos.get("title", "Desconocido")
            resumen = datos.get("extract", "Sin resumen.")
            return f"Tema Sorpresa: {titulo}\n\n{resumen}"
        return "Error al buscar un artículo aleatorio."

if __name__ == "__main__":
    mcp.run()