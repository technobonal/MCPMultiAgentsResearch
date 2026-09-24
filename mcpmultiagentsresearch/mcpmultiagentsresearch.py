"""
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
from seguridad.oauth import _verificar_sesion, COOKIE_NAME, con_autenticacion

# Activa el firewall
instalar_guardia_red()

# --- Recursos compartidos ---
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
    imagenes: list[str] = []
    eventos_investigador: list[str] = []
    eventos_multimedia: list[str] = []
    eventos_aprendizaje: list[str] = []

    sesion_cookie: str = rx.Cookie(name=COOKIE_NAME)
    usuario_autenticado: bool = False

    async def verificar_autenticacion(self):
        """Verifica la validez de la sesión antes de mostrar el chat."""
        email_valido = _verificar_sesion(self.sesion_cookie)

        if not email_valido:
            self.usuario_autenticado = False
            yield rx.redirect("/login")
            return

        self.usuario_autenticado = True

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
        print("[chat] === enviar_mensaje FUE LLAMADO ===")

        async with self:
            if not self.entrada.strip():
                print("[chat] entrada vacía → salgo")
                return

            texto_usuario = self.entrada
            print(f"[chat] texto: {texto_usuario[:80]}")
            self.mensajes.append(Mensaje(rol="user", contenido=texto_usuario))
            self.entrada = ""
            self.cargando = True
            self.eventos_investigador = []
            self.eventos_multimedia = []
            self.eventos_aprendizaje = []
            self.imagenes = []

        try:
            thread_id = self.router.session.client_token
            print(f"[chat] thread_id: {thread_id}")
            print("[chat] obteniendo grafo...")
            grafo = await _obtener_grafo()
            print("[chat] grafo OK")
        except Exception as e:
            print(f"[chat] ERROR al obtener grafo: {type(e).__name__}: {e}")
            async with self:
                self.mensajes.append(
                    Mensaje(rol="assistant", contenido=f"Error al iniciar: {e}")
                )
                self.cargando = False
            return

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

        try:
            print("[chat] arrancando astream_events...")
            async for evento in grafo.astream_events(
                estado_inicial, config=config, version="v2"
            ):
                tipo = evento.get("event")
                nombre_nodo = evento.get("metadata", {}).get("langgraph_node")
                agente = self._agente_de_nodo(nombre_nodo) if nombre_nodo else None

                if tipo == "on_tool_start" and agente:
                    async with self:
                        self._agregar_evento(
                            agente, f"🔧 llamando a {evento.get('name', 'tool')}"
                        )

                if tipo == "on_tool_end" and agente:
                    async with self:
                        self._agregar_evento(
                            agente, f"✅ {evento.get('name', 'tool')} respondió"
                        )

                if (
                    tipo == "on_chain_end"
                    and nombre_nodo == "multimedia_padded"
                    and evento.get("name") == "multimedia_padded"
                ):
                    salida = evento.get("data", {}).get("output", {})
                    if isinstance(salida, dict):
                        resultado_media = salida.get("resultado_multimedia") or {}
                        urls_imagenes = resultado_media.get("imagenes", [])

                if (
                    tipo == "on_chain_start"
                    and nombre_nodo == "aprendizaje_fase1"
                    and evento.get("name") == "aprendizaje_fase1"
                ):
                    async with self:
                        self._agregar_evento(
                            "aprendizaje", "🧠 revisando base de datos..."
                        )

                if (
                    tipo == "on_chain_end"
                    and nombre_nodo == "aprendizaje_fase1"
                    and evento.get("name") == "aprendizaje_fase1"
                ):
                    salida = evento.get("data", {}).get("output", {})
                    if isinstance(salida, dict):
                        async with self:
                            if salida.get("resultado_aprendizaje"):
                                self._agregar_evento(
                                    "aprendizaje",
                                    "✅ encontrado en memoria, no hace falta buscar de nuevo",
                                )
                            else:
                                self._agregar_evento(
                                    "aprendizaje",
                                    "sin coincidencias, esperando al Investigador",
                                )

                if (
                    tipo == "on_chain_start"
                    and nombre_nodo == "aprendizaje_fase2"
                    and evento.get("name") == "aprendizaje_fase2"
                ):
                    async with self:
                        self._agregar_evento(
                            "aprendizaje", "preparando la explicación..."
                        )

                if (
                    tipo == "on_chain_end"
                    and nombre_nodo == "aprendizaje_fase2"
                    and evento.get("name") == "aprendizaje_fase2"
                ):
                    async with self:
                        self._agregar_evento("aprendizaje", "💾 guardado en memoria")

                if (
                    tipo == "on_chain_end"
                    and nombre_nodo == "respuesta_final"
                    and evento.get("name") == "respuesta_final"
                ):
                    salida = evento.get("data", {}).get("output", {})
                    if isinstance(salida, dict):
                        respuesta_final = salida.get(
                            "respuesta_final", respuesta_final
                        )

            print("[chat] astream_events terminó OK")

        except Exception as e:
            print(f"[chat] ERROR en grafo: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            respuesta_final = (
                "Hubo un problema procesando tu consulta. Probá de nuevo en un momento."
            )
            async with self:
                self._agregar_evento("investigador", f"⚠️ Error: {type(e).__name__}")

        async with self:
            self.mensajes.append(
                Mensaje(rol="assistant", contenido=respuesta_final)
            )
            self.imagenes = urls_imagenes
            self.cargando = False
            print("[chat] respuesta enviada al UI")


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
            rx.box(
                width="8px",
                height="8px",
                border_radius="50%",
                background=color_acento,
            ),
            rx.text(
                titulo,
                font_family=FUENTE_DISPLAY,
                font_weight="600",
                color=COLOR_TINTA,
                font_size="0.95rem",
            ),
            spacing="2",
            align="center",
        ),
        rx.vstack(
            rx.foreach(
                eventos,
                lambda e: rx.text(e, font_size="0.82rem", color=COLOR_TINTA_SUAVE),
            ),
            spacing="1",
            align_items="flex-start",
            padding_top="0.5rem",
        ),
        background=COLOR_PANEL,
        border_left=f"2px solid {color_acento}",
        border_radius="10px",
        padding="1rem",
        width="100%",
    )


def index() -> rx.Component:
    dashboard = rx.box(
        rx.box(
            position="fixed",
            inset="0",
            z_index="0",
            background_image="radial-gradient(circle, #6d5bff22 1px, transparent 1px)",
            background_size="22px 22px",
        ),
        rx.hstack(
            rx.vstack(
                rx.heading(
                    "MCPMULTIAgentsResearch",
                    font_family=FUENTE_DISPLAY,
                    font_weight="700",
                    font_size="1.4rem",
                    background=GRADIENTE,
                    background_clip="text",
                    color="transparent",
                ),
                rx.vstack(
                    rx.foreach(EstadoChat.mensajes, burbuja_mensaje),
                    spacing="3",
                    width="100%",
                    overflow_y="auto",
                    flex="1",
                    padding="1rem 0",
                ),
                rx.cond(
                    EstadoChat.imagenes.length() > 0,
                    rx.hstack(
                        rx.foreach(
                            EstadoChat.imagenes,
                            lambda url: rx.image(
                                src=url, width="120px", border_radius="8px"
                            ),
                        ),
                        spacing="2",
                    ),
                ),
                rx.hstack(
                    rx.input(
                        value=EstadoChat.entrada,
                        on_change=EstadoChat.set_entrada,
                        placeholder="Preguntá sobre cualquier tema de Wikipedia...",
                        background=COLOR_PANEL,
                        color=COLOR_TINTA,
                        border=f"1px solid {COLOR_VIOLETA}55",
                        flex="1",
                    ),
                    rx.button(
                        "Enviar",
                        on_click=EstadoChat.enviar_mensaje,
                        background=GRADIENTE,
                        color="white",
                        font_family=FUENTE_DISPLAY,
                        is_loading=EstadoChat.cargando,
                    ),
                    width="100%",
                ),
                width="60%",
                height="100vh",
                padding="1.5rem",
                spacing="3",
            ),
            rx.vstack(
                rx.text(
                    "Actividad en vivo",
                    font_family=FUENTE_DISPLAY,
                    font_weight="600",
                    color=COLOR_TINTA,
                    font_size="1rem",
                ),
                panel_agente(
                    "Investigador", EstadoChat.eventos_investigador, COLOR_MAGENTA
                ),
                panel_agente(
                    "Multimedia", EstadoChat.eventos_multimedia, "#c34bff"
                ),
                panel_agente(
                    "Aprendizaje", EstadoChat.eventos_aprendizaje, COLOR_VIOLETA
                ),
                width="40%",
                height="100vh",
                padding="1.5rem",
                spacing="3",
                background=f"{COLOR_PANEL}88",
                overflow_y="auto",
            ),
            width="100%",
            spacing="0",
            position="relative",
            z_index="1",
        ),
        background=COLOR_FONDO,
        min_height="100vh",
        font_family=FUENTE_BODY,
    )

    return rx.cond(
        EstadoChat.usuario_autenticado,
        dashboard,
        rx.box(
            rx.text(
                "Verificando credenciales...",
                color=COLOR_TINTA,
                font_family=FUENTE_DISPLAY,
            ),
            display="flex",
            align_items="center",
            justify_content="center",
            min_height="100vh",
            background=COLOR_FONDO,
        ),
    )


def login() -> rx.Component:
    return rx.box(
        rx.box(
            rx.heading(
                "MCPMULTIAgentsResearch",
                font_family=FUENTE_DISPLAY,
                font_weight="700",
                font_size="1.6rem",
                background=GRADIENTE,
                background_clip="text",
                color="transparent",
                margin_bottom="0.5rem",
            ),
            rx.text(
                "Bienvenido de nuevo",
                color=COLOR_TINTA,
                font_size="1.1rem",
                margin_bottom="2rem",
            ),
            rx.el.a(
                rx.box(
                    "Continuar con Google",
                    background=GRADIENTE,
                    color="white",
                    font_family=FUENTE_DISPLAY,
                    padding="0.8rem 1.5rem",
                    border_radius="10px",
                    text_align="center",
                ),
                href="http://localhost:8000/auth/google/start",
                width="100%",
                style={"text_decoration": "none", "display": "block"},
            ),
            background=COLOR_PANEL,
            border=f"1px solid {COLOR_VIOLETA}33",
            border_radius="14px",
            padding="2.5rem",
            width="360px",
        ),
        display="flex",
        align_items="center",
        justify_content="center",
        min_height="100vh",
        background=COLOR_FONDO,
    )


app = rx.App(
    style={"font_family": FUENTE_BODY, "background": COLOR_FONDO},
    stylesheets=[
        "https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500&display=swap",
    ],
    api_transformer=con_autenticacion,
)
app.add_page(
    index,
    title="MCPMULTIAgentsResearch",
    on_load=EstadoChat.verificar_autenticacion,
)
app.add_page(login, route="/login", title="Iniciar sesión")