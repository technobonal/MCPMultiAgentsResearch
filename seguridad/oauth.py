"""
seguridad/oauth.py

Login con Google OAuth 2.0 restringido a un solo email (ALLOWED_EMAIL).
El gateo de la página "/" se hace del lado de Reflex (rx.Cookie +
on_load) -- acá solo se resuelven las 3 rutas de auth.
"""

from __future__ import annotations

import os

from authlib.integrations.starlette_client import OAuth
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import RedirectResponse
from starlette.routing import Route
from starlette.middleware.sessions import SessionMiddleware

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
ALLOWED_EMAIL = os.getenv("ALLOWED_EMAIL", "")
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

COOKIE_NAME = "mcpma_session"
COOKIE_MAX_AGE = 60 * 60 * 24 * 7

_serializer = URLSafeTimedSerializer(SESSION_SECRET_KEY, salt="oauth-session")

oauth = OAuth()
oauth.register(
    name="google",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)


def _firmar_sesion(email: str) -> str:
    return _serializer.dumps({"email": email})


def _verificar_sesion(cookie_valor: str | None) -> str | None:
    if not cookie_valor:
        return None
    try:
        datos = _serializer.loads(cookie_valor, max_age=COOKIE_MAX_AGE)
        return datos.get("email")
    except (BadSignature, SignatureExpired):
        return None


async def ruta_login_start(request: Request):
    redirect_uri = str(request.base_url).rstrip("/") + "/api/auth/callback"
    return await oauth.google.authorize_redirect(request, redirect_uri)


async def ruta_callback(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
        userinfo = token.get("userinfo") or {}
        email = userinfo.get("email", "")

        if not email or email.lower() != ALLOWED_EMAIL.lower():
            return RedirectResponse(url=f"{FRONTEND_URL}/login?error=no_autorizado")

        token_firmado = _firmar_sesion(email)

        respuesta = RedirectResponse(url=f"{FRONTEND_URL}/", status_code=302)
        respuesta.set_cookie(
            key=COOKIE_NAME,
            value=token_firmado,
            max_age=COOKIE_MAX_AGE,
            path="/",
            httponly=False,   # necesario para que Reflex lea la cookie
            samesite="lax",
        )
        return respuesta

    except Exception:
        return RedirectResponse(url=f"{FRONTEND_URL}/login?error=callback_failed")


async def ruta_logout(request: Request):
    respuesta = RedirectResponse(url=f"{FRONTEND_URL}/login")
    respuesta.delete_cookie(COOKIE_NAME, path="/")
    return respuesta


_auth_app = Starlette(routes=[
    Route("/auth/google/start", ruta_login_start),
    Route("/api/auth/callback", ruta_callback, name="ruta_callback"),
    Route("/logout", ruta_logout),
])

_auth_app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET_KEY,
    max_age=COOKIE_MAX_AGE,
    same_site="lax",
)

_RUTAS_AUTH = {"/auth/google/start", "/api/auth/callback", "/logout"}


def con_autenticacion(asgi_app):
    async def wrapped(scope, receive, send):
        if scope["type"] == "http" and scope["path"] in _RUTAS_AUTH:
            await _auth_app(scope, receive, send)
            return
        await asgi_app(scope, receive, send)

    return wrapped