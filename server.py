import subprocess
import time
import logging
import httpx
import atexit
from huggingface_hub import snapshot_download


MODEL = "mlx-community/Qwen3.6-35B-A3B-4bit"
SERVER_URL = "http://localhost:8000/v1"
_proc = None

logger = logging.getLogger(__name__)


# --- SETUP SEPERATE LOGGING FOR SERVER OUTPUT ---
logger_server = logging.getLogger("server")
logger_server.setLevel(logging.INFO)

file_handler = logging.FileHandler("server.log", mode='w')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger_server.addHandler(file_handler)
logger_server.propagate = False  # Prevent log messages from being propagated to the root logger


def start():
    global _proc

    # skip if already running
    try:
        if httpx.get("http://localhost:8000/health").status_code == 200:
            logger.info("Server is already running.")
            return
    except httpx.RequestError:
        pass



    model_path = snapshot_download(MODEL, local_files_only=True)
    logger.info("Starting server...")

    #open file for subprocess to write to
    server_log_file = open("server.log", "w")

    _proc = subprocess.Popen(
        [
            "mlx-openai-server", "launch",
            "--model-type", "lm",
            "--model-path", model_path      ,
            "--reasoning-parser", "qwen3",
            "--enable-auto-tool-choice",
            "--context-length", "16384",
        ],
        stdout=server_log_file,
        stderr=subprocess.STDOUT,
    )

    def cleanup():
        if _proc:
            _proc.terminate()
        server_log_file.close()

    atexit.register(cleanup)
    _wait_for_server()
    


def _wait_for_server(timeout: int = 180):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            if httpx.get("http://localhost:8000/health").status_code == 200:
                logger.info("Server is ready.")
                return
        except httpx.RequestError:
            elapsed = int(time.time() - start_time)
            logger.info(f"Waiting for server... (elapsed: {elapsed}s)")
        time.sleep(2)
    raise TimeoutError(f"Server did not become ready within {timeout} seconds.")