"""
app.py

Interfaz Reflex: chat a la izquierda, actividad en vivo de los 3
agentes a la derecha. Estética: fondo casi negro con tinte violeta,
degradé magenta->violeta reservado al wordmark y al estado, fondo con
puntos sutiles evocando un grafo de conocimiento.
"""

from __future__ import annotations

import asyncio

import reflex as rx
from dataclasses import dataclass
from langchain_core.messages import HumanMessage

from grafo import construir_grafo
from seguridad.guardia_red import instalar_guardia_red
from seguridad.rate_limiter import RateLimiter

instalar_guardia_red()

# --- Recursos compartidos, una sola instancia para toda la app ---
_rate_limiter = RateLimiter()
_grafo = None
_grafo_lock = asyncio.Lock()


async def _obtener_grafo():
    global _grafo
    async with _grafo_lock:
        if _grafo is None:
            _grafo = await construir_grafo(_rate_limiter)
    return _grafo


# --- Design tokens ---
COLOR_FONDO = "#0a0710"
COLOR_PANEL = "#140d21"
COLOR_TINTA = "#f2eefb"
COLOR_TINTA_SUAVE = "#9c93b8"
COLOR_MAGENTA = "#ff2e88"
COLOR_VIOLETA = "#6d5bff"
GRADIENTE = f"linear-gradient(135deg, {COLOR_MAGENTA}, {COLOR_VIOLETA})"
FUENTE_DISPLAY = "'Space Grotesk', sans-serif"
FUENTE_BODY = "'Inter', sans-serif"


@dataclass
class Mensaje:
    rol: str
    contenido: str


class EstadoChat(rx.State):
    mensajes: list[Mensaje] = []
    entrada: str = ""
    cargando: bool = False
    imagenes: list[str] = []          # <-- AGREGAR ACÁ

    eventos_investigador: list[str] = []
    eventos_multimedia: list[str] = []
    eventos_aprendizaje: list[str] = []

    def set_entrada(self, valor: str):
        self.entrada = valor

    def _agente_de_nodo(self, nombre_nodo: str) -> str | None:
        return {
            "investigador": "investigador",
            "multimedia_padded": "multimedia",
            "aprendizaje_fase1": "aprendizaje",
            "aprendizaje_fase2": "aprendizaje",
        }.get(nombre_nodo)

    def _agregar_evento(self, agente: str, texto: str):
        if agente == "investigador":
            self.eventos_investigador.append(texto)
        elif agente == "multimedia":
            self.eventos_multimedia.append(texto)
        elif agente == "aprendizaje":
            self.eventos_aprendizaje.append(texto)

    @rx.event(background=True)
    async def enviar_mensaje(self):
        async with self:
            if not self.entrada.strip():
                return
            texto_usuario = self.entrada
            self.mensajes.append(Mensaje(rol="user", contenido=texto_usuario))
            self.entrada = ""
            self.cargando = True
            self.eventos_investigador = []
            self.eventos_multimedia = []
            self.eventos_aprendizaje = []
            self.imagenes = []

        thread_id = self.router.session.client_token
        grafo = await _obtener_grafo()

        estado_inicial = {
            "mensajes": [HumanMessage(content=texto_usuario)],
            "thread_id": thread_id,
            "input_sanitizado": None,
            "tema_consulta": None,
            "tipo_solicitud": None,
            "resultado_investigador": None,
            "resultado_multimedia": None,
            "resultado_aprendizaje": None,
            "pasos_investigacion": [],
            "fuentes": None,
            "respuesta_final": None,
            "estado": "recibido",
            "error": None,
        }
        config = {"configurable": {"thread_id": thread_id}}
        respuesta_final = "No se encontró información."
        urls_imagenes: list[str] = []

        async for evento in grafo.astream_events(estado_inicial, config=config, version="v2"):
            tipo = evento.get("event")
            nombre_nodo = evento.get("metadata", {}).get("langgraph_node")
            agente = self._agente_de_nodo(nombre_nodo) if nombre_nodo else None

            if tipo == "on_tool_start" and agente:
                async with self:
                    self._agregar_evento(agente, f"🔧 llamando a {evento.get('name', 'tool')}")

            if tipo == "on_tool_end" and agente:
                async with self:
                    self._agregar_evento(agente, f"✅ {evento.get('name', 'tool')} respondió")
            if tipo == "on_chain_end" and nombre_nodo == "multimedia_padded" and evento.get("name") == "multimedia_padded":
                salida = evento.get("data", {}).get("output", {})
                if isinstance(salida, dict):
                    resultado_media = salida.get("resultado_multimedia") or {}
                    urls_imagenes = resultado_media.get("imagenes", [])

            if tipo == "on_chain_start" and nombre_nodo == "aprendizaje_fase1" and evento.get("name") == "aprendizaje_fase1":
                async with self:
                    self._agregar_evento("aprendizaje", "🧠 revisando base de datos...")

            if tipo == "on_chain_end" and nombre_nodo == "aprendizaje_fase1" and evento.get("name") == "aprendizaje_fase1":
                salida = evento.get("data", {}).get("output", {})
                if isinstance(salida, dict):
                    async with self:
                        if salida.get("resultado_aprendizaje"):
                            self._agregar_evento("aprendizaje", "✅ encontrado en memoria, no hace falta buscar de nuevo")
                        else:
                            self._agregar_evento("aprendizaje", "sin coincidencias, esperando al Investigador")

            if tipo == "on_chain_start" and nombre_nodo == "aprendizaje_fase2" and evento.get("name") == "aprendizaje_fase2":
                async with self:
                    self._agregar_evento("aprendizaje", "preparando la explicación...")

            if tipo == "on_chain_end" and nombre_nodo == "aprendizaje_fase2" and evento.get("name") == "aprendizaje_fase2":
                async with self:
                    self._agregar_evento("aprendizaje", "💾 guardado en memoria")
            # --- FIN BLOQUES NUEVOS --- 
            if tipo == "on_chain_end" and nombre_nodo == "respuesta_final" and evento.get("name") == "respuesta_final":
                salida = evento.get("data", {}).get("output", {})
                if isinstance(salida, dict):
                    respuesta_final = salida.get("respuesta_final", respuesta_final)

        async with self:
            self.mensajes.append(Mensaje(rol="assistant", contenido=respuesta_final))
            self.imagenes = urls_imagenes
            self.cargando = False


def burbuja_mensaje(msg: Mensaje) -> rx.Component:
    es_usuario = msg.rol == "user"
    return rx.box(
        rx.cond(
            es_usuario,
            rx.text(msg.contenido, color=COLOR_TINTA),
            rx.markdown(msg.contenido, color=COLOR_TINTA),
        ),
        background=rx.cond(es_usuario, COLOR_PANEL, "transparent"),
        border=rx.cond(es_usuario, "none", f"1px solid {COLOR_VIOLETA}33"),
        border_radius="12px",
        padding="0.9rem 1.1rem",
        max_width="80%",
        align_self=rx.cond(es_usuario, "flex-end", "flex-start"),
    )


def panel_agente(titulo: str, eventos: rx.Var, color_acento: str) -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.box(width="8px", height="8px", border_radius="50%", background=color_acento),
            rx.text(titulo, font_family=FUENTE_DISPLAY, font_weight="600", color=COLOR_TINTA, font_size="0.95rem"),
            spacing="2", align="center",
        ),
        rx.vstack(
            rx.foreach(eventos, lambda e: rx.text(e, font_size="0.82rem", color=COLOR_TINTA_SUAVE)),
            spacing="1", align_items="flex-start", padding_top="0.5rem",
        ),
        background=COLOR_PANEL, border_left=f"2px solid {color_acento}",
        border_radius="10px", padding="1rem", width="100%",
    )


def index() -> rx.Component:
    return rx.box(
        rx.box(
            position="fixed", inset="0", z_index="0",
            background_image="radial-gradient(circle, #6d5bff22 1px, transparent 1px)",
            background_size="22px 22px",
        ),
        rx.hstack(
            rx.vstack(
                rx.heading(
                    "MCPMULTIAgentsResearch",
                    font_family=FUENTE_DISPLAY, font_weight="700", font_size="1.4rem",
                    background=GRADIENTE, background_clip="text", color="transparent",
                ),
                rx.vstack(
                    rx.foreach(EstadoChat.mensajes, burbuja_mensaje),
                    spacing="3", width="100%", overflow_y="auto", flex="1", padding="1rem 0",
                ),
                rx.cond(
                    EstadoChat.imagenes.length() > 0,
                    rx.hstack(
                        rx.foreach(
                            EstadoChat.imagenes,
                            lambda url: rx.image(src=url, width="120px", border_radius="8px")
                        ),
                        spacing="2",
                    ),
                ),
                rx.hstack(
                    rx.input(
                        value=EstadoChat.entrada, on_change=EstadoChat.set_entrada,
                        placeholder="Preguntá sobre cualquier tema de Wikipedia...",
                        background=COLOR_PANEL, color=COLOR_TINTA,
                        border=f"1px solid {COLOR_VIOLETA}55", flex="1",
                    ),
                    rx.button(
                        "Enviar", on_click=EstadoChat.enviar_mensaje,
                        background=GRADIENTE, color="white",
                        font_family=FUENTE_DISPLAY, is_loading=EstadoChat.cargando,
                    ),
                    width="100%",
                ),
                width="60%", height="100vh", padding="1.5rem", spacing="3",
            ),
            rx.vstack(
                rx.text("Actividad en vivo", font_family=FUENTE_DISPLAY, font_weight="600",
                        color=COLOR_TINTA, font_size="1rem"),
                panel_agente("Investigador", EstadoChat.eventos_investigador, COLOR_MAGENTA),
                panel_agente("Multimedia", EstadoChat.eventos_multimedia, "#c34bff"),
                panel_agente("Aprendizaje", EstadoChat.eventos_aprendizaje, COLOR_VIOLETA),
                width="40%", height="100vh", padding="1.5rem", spacing="3",
                background=f"{COLOR_PANEL}88", overflow_y="auto",
            ),
            width="100%", spacing="0", position="relative", z_index="1",
        ),
        background=COLOR_FONDO, min_height="100vh", font_family=FUENTE_BODY,
    )


app = rx.App(
    style={"font_family": FUENTE_BODY, "background": COLOR_FONDO},
    stylesheets=[
        "https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500&display=swap",
    ],
)
app.add_page(index, title="MCPMULTIAgentsResearch")