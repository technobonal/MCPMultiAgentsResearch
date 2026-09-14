"""
seguridad/guardia_red.py

Allowlist de red a nivel de proceso principal (app.py). Bloquea toda
conexión saliente que no sea a Groq o Hugging Face -- servidor_wiki.py
corre como subproceso separado y no se ve afectado por esto (necesita
hablarle a Wikipedia directo, por diseño).

Funciona parcheando socket.getaddrinfo: casi todas las librerías HTTP
(httpx, requests, urllib3) pasan por ahí para resolver el hostname
antes de conectar -- bloquear la resolución corta la conexión temprano,
con un error claro, sin tener que inspeccionar IPs en connect().
"""

from __future__ import annotations

import logging
import socket

logger = logging.getLogger("mcpmultiagentsresearch.guardia_red")

# Sufijos de host permitidos -- un hostname pasa si termina en alguno
# de estos (permite subdominios, ej. cdn-lfs.huggingface.co).
ALLOWLIST_SUFIJOS = (
    "api.groq.com",
    "huggingface.co",
    "hf.co",
)

# Hosts locales -- necesarios para que el propio servidor de Reflex
# (frontend + backend) pueda bindear/resolverse a sí mismo. No son
# conexiones salientes a internet, así que no corresponde al alcance
# del allowlist de red externa.
_HOSTS_LOCALES = ("localhost", "127.0.0.1", "0.0.0.0", "::1")
_getaddrinfo_original = socket.getaddrinfo
_instalada = False


def _host_permitido(host: str | None) -> bool:
    if host is None:
        return True
    host = host.lower()
    if host in _HOSTS_LOCALES:
        return True
    return any(host == sufijo or host.endswith("." + sufijo) for sufijo in ALLOWLIST_SUFIJOS)


def _getaddrinfo_filtrado(host, *args, **kwargs):
    if not _host_permitido(host):
        logger.warning(f"[guardia_red] Conexión bloqueada: intento de resolver '{host}'")
        raise PermissionError(
            f"guardia_red: '{host}' no está en el allowlist de red "
            f"({', '.join(ALLOWLIST_SUFIJOS)})"
        )
    return _getaddrinfo_original(host, *args, **kwargs)


def instalar_guardia_red() -> None:
    """
    Se llama UNA SOLA VEZ, al arrancar app.py, antes de crear cualquier
    agente o LLM. Parchea socket.getaddrinfo a nivel de proceso -- no
    afecta a servidor_wiki.py (proceso aparte).
    """
    global _instalada
    if _instalada:
        return
    socket.getaddrinfo = _getaddrinfo_filtrado
    _instalada = True
    logger.info(f"[guardia_red] Instalada. Allowlist: {ALLOWLIST_SUFIJOS}")


if __name__ == "__main__":
    instalar_guardia_red()

    import httpx

    print("Probando conexión permitida (Groq)...")
    try:
        httpx.get("https://api.groq.com", timeout=5)
        print("✅ Conexión a api.groq.com permitida (o falló por otro motivo, no por el guardia)")
    except PermissionError as e:
        print(f"❌ Bloqueada incorrectamente: {e}")
    except Exception as e:
        print(f"(Conexión intentada, falló por otra razón esperable: {e})")

    print("\nProbando conexión NO permitida (ejemplo.com)...")
    try:
        httpx.get("https://ejemplo.com", timeout=5)
        print("❌ FALLO DE SEGURIDAD: se permitió una conexión que debería estar bloqueada")
    except PermissionError as e:
        print(f"✅ Bloqueada correctamente: {e}")