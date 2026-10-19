"""Start the web app on http://<pc>:8000 and https://<pc>:8443 in one process (shared models).

The phone uploads photos over plain http on the http port; the https port is optional.
First visit on the phone shows a certificate warning (self-signed): Advanced -> continue.
"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import uvicorn

from grade_eval import phone
from grade_eval.server import app


async def main():
    cert, key = phone.ensure_cert()
    http = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=phone.HTTP_PORT))
    https = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=phone.HTTPS_PORT, ssl_certfile=str(cert), ssl_keyfile=str(key)))
    https.install_signal_handlers = lambda: None            # Ctrl+C is handled once, by the http server

    async def run_http():
        await http.serve()
        https.should_exit = True

    ip = phone.lan_ip()
    print(f"\n  On this PC:    http://localhost:{phone.HTTP_PORT}\n  Phone upload: http://{ip}:{phone.HTTP_PORT}  (same Wi-Fi)\n")
    await asyncio.gather(run_http(), https.serve())


if __name__ == "__main__":
    asyncio.run(main())
