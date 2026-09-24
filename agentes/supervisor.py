"""
agentes/supervisor.py

Nodo orquestador del grafo. Responsabilidades:
1. Clasifica la intención del usuario UNA SOLA VEZ, vía structured
   output (IntencionOut) -- evita que cada agente tenga que
   re-derivar el tema por su cuenta.
2. Deja tema_consulta en el AgentState para que el fan-out lo lea
   directo, sin que el supervisor tenga que "pasárselo" a cada agente
   a mano (eso lo resuelve grafo.py con los edges).

El supervisor NO ejecuta tools ni llama a Wikipedia -- solo clasifica
y decide el routing. La ejecución real vive en cada agente.

Usa los últimos turnos del historial (state["mensajes"]) para poder
resolver referencias al contexto previo -- ej. "y sus fotos" después
de haber hablado de "el león" tiene que clasificar tema_consulta como
"león", no como algo vacío o literal "sus fotos".

NUEVO -- tercer valor de tipo_solicitud: "explicacion_sin_imagen".
Antes solo existían "informacion_general" (siempre traía imagen si
Multimedia corría) y "solo_imagenes" -- no había forma de pedir
explícitamente texto SIN imagen, así que un pedido puramente
explicativo como "qué es una bisagra" terminaba trayendo una foto
igual, porque el fan-out (ver grafo.py) disparaba a Multimedia salvo
que el pedido fuera "solo_imagenes". La distinción clave para el LLM:
la AUSENCIA de mención a imágenes sigue siendo "informacion_general"
(comportamiento de siempre, trae texto + imagen) -- el nuevo valor es
solo para cuando el usuario NIEGA explícitamente la imagen ("sin
fotos", "no quiero imagen", "solo la explicación").
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agent_state import AgentState
from llm_client import get_llm
from seguridad.rate_limiter import RateLimiter

_SYSTEM_PROMPT = """Sos el supervisor de un sistema de investigación sobre Wikipedia.
Tu único trabajo es leer el pedido del usuario y clasificarlo en una
estructura, NO respondés directamente al usuario.

Vas a recibir el historial reciente de la conversación. Usalo para
resolver referencias ambiguas al último turno del usuario -- por
ejemplo, si antes se habló de "el león" y el usuario ahora escribe
"y sus fotos", el tema_consulta tiene que ser "león", no "sus fotos"
ni algo vacío. Siempre clasificá el ÚLTIMO mensaje del usuario, usando
los turnos anteriores solo como contexto para resolver esas referencias.

Extraé:
- tema_consulta: el tema o entidad concreta que hay que investigar.
  Sacá SOLO muletillas conversacionales (ej. "contame sobre", "quiero
  saber de", "che", "dale"). NUNCA saques palabras que desambiguan el
  tema cuando el nombre es ambiguo -- ej. "planeta Mercurio" tiene que
  quedar como "planeta Mercurio", no como "Mercurio" a secas (que
  podría confundirse con el elemento químico o el dios romano). Ante
  la duda, dejá el calificador.
- tipo_solicitud, una de estas TRES opciones:
    "informacion_general" -> el CASO POR DEFECTO. El usuario pide
      explicación/info sobre el tema y NO dice nada sobre imágenes,
      ni a favor ni en contra (ej. "qué es una bisagra", "contame
      sobre el león", "explicame la fotosíntesis"). Este caso SÍ trae
      imagen además del texto -- es el comportamiento normal.
    "solo_imagenes" -> el usuario pide ESPECÍFICAMENTE imágenes/fotos
      del tema, sin pedir explicación (ej. "mostrame fotos de leones",
      "quiero ver imágenes de una bisagra", "una foto del tema").
    "explicacion_sin_imagen" -> el usuario pide explicación y NIEGA
      EXPLÍCITAMENTE que quiere imagen (ej. "explicame qué es una
      bisagra, sin fotos", "solo texto, no quiero imágenes", "explicá
      sin mostrar fotos"). OJO: esto NO es lo mismo que
      "informacion_general" -- la clave es que el usuario tiene que
      RECHAZAR la imagen de forma explícita, no simplemente no
      mencionarla. Si no hay una negación clara, es
      "informacion_general".
- urgencia: opcional, solo si el usuario la expresa explícitamente
  (ej. "necesito esto ya"); en general dejarlo vacío.

Si el pedido no tiene un tema claro ni hay contexto previo que lo
resuelva, hacé tu mejor inferencia del tema más probable a partir del
texto."""


class IntencionOut(BaseModel):
    tema_consulta: str = Field(description="Tema normalizado a investigar")
    tipo_solicitud: Literal[
        "informacion_general", "solo_imagenes", "explicacion_sin_imagen"
    ] = Field(description="Qué tipo de resultado espera el usuario")
    urgencia: str | None = Field(
        default=None, description="Urgencia expresada explícitamente por el usuario, si la hay"
    )


def construir_nodo_supervisor(rate_limiter: RateLimiter):
    async def nodo_supervisor(state: AgentState) -> dict:
        print("[DEBUG] entrando a nodo_supervisor", flush=True)

        historial = state.get("mensajes", [])[-6:]
        mensajes_historial = [
            {"role": "user" if m.type == "human" else "assistant", "content": m.content}
            for m in historial
            if m.type in ("human", "ai")
        ]

        mensajes = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *mensajes_historial,
        ]

        print("[DEBUG] a punto de llamar a get_llm...", flush=True)
        llm = get_llm(mensajes, rate_limiter, temperature=0.0)
        print("[DEBUG] get_llm devolvió OK", flush=True)
        clasificador = llm.with_structured_output(IntencionOut)
        print("[DEBUG] with_structured_output armado OK", flush=True)

        print("[DEBUG] a punto de llamar a Groq (supervisor)...", flush=True)
        intencion: IntencionOut = await clasificador.ainvoke(mensajes)
        print(f"[DEBUG] supervisor clasificó: tema_consulta={intencion.tema_consulta!r}, tipo_solicitud={intencion.tipo_solicitud!r}", flush=True)

        return {
            "tema_consulta": intencion.tema_consulta,
            "tipo_solicitud": intencion.tipo_solicitud,
            "estado": "enrutando",
        }

    return nodo_supervisor