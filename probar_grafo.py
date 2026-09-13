import asyncio
from grafo import construir_grafo
from seguridad.rate_limiter import RateLimiter
from langchain_core.messages import HumanMessage

async def probar():
    rate_limiter = RateLimiter()
    grafo = await construir_grafo(rate_limiter)

    estado_inicial = {
        'mensajes': [HumanMessage(content='Contame sobre agujeros negros')],
        'thread_id': 'prueba-1',
        'input_sanitizado': None,
        'tema_consulta': None,
        'tipo_solicitud': None,
        'resultado_investigador': None,
        'resultado_multimedia': None,
        'resultado_aprendizaje': None,
        'pasos_investigacion': [],
        'fuentes': None,
        'respuesta_final': None,
        'estado': 'recibido',
        'error': None,
    }

    config = {'configurable': {'thread_id': 'prueba-1'}}
    resultado = await grafo.ainvoke(estado_inicial, config=config)

    print('--- RESPUESTA FINAL ---')
    print(resultado['respuesta_final'])
    print()
    print('--- FUENTES ---')
    print(resultado['fuentes'])
    print()
    print('--- PASOS DE INVESTIGACION ---')
    print(resultado['pasos_investigacion'])

asyncio.run(probar())