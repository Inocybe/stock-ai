import subprocess
import time
import httpx
import atexit


SERVER_URL = "http://localhost:8000/v1"
_proc = None

"""
First run start_server() to launch the server, 
then run wait_for_server() to wait for it to be ready.
"""

def start():
    global _proc

    # skip if already running
    try:
        if httpx.get("http://localhost:8000/health").status_code == 200:
            print("Server is already running.")
            return


def start_server():
    proc = subprocess.Popen(
        [
            "mlx-openai-server", "launch"
            "--model-type", "lm",
            "--model-path", "models--mlx-community--Qwen3.6-35B-A3B-4bit",
            "--tool-call-parser", "qwen3",
            "--reasoning-parser", "qwen3",
            "--enable-auto-tool-choise",
            "--context-length", "16384",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atexit.register(lambda: proc.terminate())
    return proc



def wait_for_server(url: str = "http://localhost:8000/health", timeout: int = 60):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = httpx.get(url)
            if response.status_code == 200:
                return True
        except httpx.RequestError:
            pass
        time.sleep(1)
    raise TimeoutError(f"Server did not become ready within {timeout} seconds.")