"""
run_async_app.py
----------------
Starts Celery and Uvicorn in a single command for local development.

Redis is expected to be running externally (Docker, system install, or
WSL on Windows). The old embedded Windows redis-server.exe has been removed
from the repo — it was a supply-chain risk and only worked on Windows.

To run Redis locally:
    Docker:  docker run -d -p 6379:6379 redis:7-alpine
    Ubuntu:  sudo apt install redis-server && sudo service redis start
    macOS:   brew install redis && brew services start redis
    Windows: Use WSL2 + Ubuntu, or Docker Desktop

Concurrency
-----------
Worker concurrency is auto-detected from CPU cores by celery_app.py.
Override with environment variable if needed:
    STATIC_ANALYSIS_CONCURRENCY=8 python run_async_app.py

Module B (LLM evaluation) worker is started automatically below.
When adding Module C (cosine similarity), add another worker:
    celery -A app.worker.celery_app worker -Q cosine_similarity --pool=prefork
"""

import subprocess
import sys
import time
import signal
import os

from app.worker.celery_app import WORKER_CONCURRENCY


def kill_process_tree(pid: int) -> None:
    """Robustly kill a process and all its children."""
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
    except Exception:
        pass


def main() -> None:
    print("Starting Module A services...")
    print(f"Worker concurrency: {WORKER_CONCURRENCY} (auto-detected from CPU cores)")
    # Redis — start embedded server if the binary exists locally
    redis_exe = os.path.join(".redis", "redis-server.exe")
    redis_proc = None
    if sys.platform == "win32" and os.path.isfile(redis_exe):
        print(f"Starting embedded Redis: {redis_exe}")
        redis_proc = subprocess.Popen(
            [redis_exe],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        time.sleep(1)   # give redis-server a moment to bind port 6379
    else:
        print("Redis: assumed running on localhost:6379 (start separately)")
    print()

    # Windows flag to prevent children from receiving Ctrl+C directly.
    cflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0

    # Celery Worker — Module A (static_analysis queue)
    # CPU-bound: Tree-sitter + Semgrep
    # Pool: solo on Windows, prefork on Linux/macOS
    if sys.platform == "win32":
        pool = "solo"
        print("Windows detected: using --pool=solo (single-threaded).")
        print("For production concurrency on Windows, use WSL2 or Docker.")
    else:
        pool = "prefork"

    print(f"Starting Celery Worker — static_analysis (pool={pool}, concurrency={WORKER_CONCURRENCY})...")
    celery_proc = subprocess.Popen(
        [
            sys.executable, "-m", "celery",
            "-A", "app.worker.celery_app",
            "worker",
            "--loglevel=info",
            f"--pool={pool}",
            f"--concurrency={WORKER_CONCURRENCY}",
            "--queues=static_analysis",
            "--hostname=worker-static@%h",
        ],
        creationflags=cflags,
    )

   
    llm_pool = "solo" if sys.platform == "win32" else "gevent"
    print(f"Starting Celery Worker — llm_evaluation (pool={llm_pool}, concurrency=1)...")
    llm_celery_proc = subprocess.Popen(
        [
            sys.executable, "-m", "celery",
            "-A", "app.worker.celery_app",
            "worker",
            "--loglevel=info",
            f"--pool={llm_pool}",
            "--concurrency=1",
            "--queues=llm_evaluation",
            "--hostname=worker-llm@%h",
        ],
        creationflags=cflags,
    )

    # Uvicorn API Server
    print("Starting Uvicorn API Server on http://localhost:8000 ...")
    uvicorn_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--reload"],
        creationflags=cflags,
    )

    # Shutdown handler
    def handle_exit(signum, frame) -> None:
        print("\nShutting down all services...")
        kill_process_tree(celery_proc.pid)
        kill_process_tree(llm_celery_proc.pid)
        kill_process_tree(uvicorn_proc.pid)
        if redis_proc is not None:
            kill_process_tree(redis_proc.pid)
        print("Shutdown complete.")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    print()
    print("All services started!")
    print(f"  API:    http://localhost:8000")
    print(f"  Docs:   http://localhost:8000/docs")
    print(f"  Workers:")
    print(f"    static_analysis : {WORKER_CONCURRENCY} concurrent")
    print(f"    llm_evaluation  : 1 (Ollama inference)")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        handle_exit(None, None)


if __name__ == "__main__":
    main()