try:  # prefer real FastAPI when available
    from fastapi import (
        FastAPI,
        WebSocket,
        HTTPException,
        Request,
        status,
        WebSocketDisconnect,
    )
    from fastapi.responses import JSONResponse
except ImportError:  # lightweight fallback used when FastAPI isn't installed
    import types

    class _DummyApp:
        def __init__(self, *a, **k):
            self._routes = []
            self._middlewares = []

        def add_api_route(self, *a, **k):
            self._routes.append((a, k))

        def add_middleware(self, *a, **k):
            self._middlewares.append((a, k))

        def middleware(self, _name):
            def decorator(func):
                return func
            return decorator

        async def __call__(self, scope, receive, send):  # pragma: no cover - stub
            pass

        def include_router(self, *a, **k):
            pass

        def on_event(self, _name):
            def decorator(func):
                return func
            return decorator

        def websocket(self, *a, **k):
            def decorator(func):
                return func
            return decorator

        def get(self, *a, **k):
            def decorator(func):
                return func
            return decorator

        def post(self, *a, **k):
            def decorator(func):
                return func
            return decorator

    FastAPI = _DummyApp

    class WebSocket:
        pass

    class WebSocketDisconnect(Exception):
        pass

    class Request:
        pass

    class HTTPException(Exception):
        pass

    status = types.SimpleNamespace(
        HTTP_200_OK=200,
        HTTP_422_UNPROCESSABLE_ENTITY=422,
    )

    class JSONResponse:
        def __init__(self, content=None, status_code=200):
            self.content = content
            self.status_code = status_code

        async def __call__(self, scope, receive, send):  # pragma: no cover - stub
            pass

__all__ = [
    "FastAPI",
    "WebSocket",
    "WebSocketDisconnect",
    "HTTPException",
    "Request",
    "status",
    "JSONResponse",
]
