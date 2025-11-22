from flask import request, Response
from functools import wraps
import os

USERNAME = os.getenv("BASIC_AUTH_USERNAME", "lena")
PASSWORD = os.getenv("BASIC_AUTH_PASSWORD", "lena123")

def require_auth(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        auth = request.authorization
        if not auth or auth.username != USERNAME or auth.password != PASSWORD:
            return Response(
                "Authentication required.",
                401,
                {"WWW-Authenticate": 'Basic realm="Login Required"'},
            )
        return fn(*args, **kwargs)
    return wrapper
