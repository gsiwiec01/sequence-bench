import json
import math

from starlette.responses import JSONResponse

def _sanitize(obj):
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj

    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]

    return obj


class NaNSafeJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(
            _sanitize(content),
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")
