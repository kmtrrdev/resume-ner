from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from core.config import settings

class SecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 1. Size restriction
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > settings.MAX_UPLOAD_SIZE:
            return JSONResponse(
                status_code=413, 
                content={"detail": f"Payload too large. Max size is {settings.MAX_UPLOAD_SIZE} bytes."}
            )

        # 2. Simple API Key protection for specific routes
        if request.url.path.startswith("/parse"):
            api_key = request.headers.get("X-API-Key")
            if api_key and api_key != settings.API_KEY:
                return JSONResponse(status_code=401, content={"detail": "Invalid API Key"})
                
        response = await call_next(request)
        return response
