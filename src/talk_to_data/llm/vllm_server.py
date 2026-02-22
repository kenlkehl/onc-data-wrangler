"""Auto-managed vLLM server subprocess lifecycle manager."""

import logging
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_HEALTH_CHECK_INTERVAL = 2.0
_HEALTH_CHECK_TIMEOUT = 600.0


@dataclass
class _ServerProcess:
    """Tracks a single vLLM server subprocess."""
    gpus: list[int]
    port: int
    proc: subprocess.Popen
    base_url: str


def _check_gpu_ids(gpus: list[int]):
    """Validate that requested GPU IDs exist on the system."""
    try:
        result = subprocess.run(
            ("nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"),
            capture_output=True, text=True, check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise RuntimeError("nvidia-smi not found or failed. Cannot validate GPU IDs.")

    available = set()
    for line in result.stdout.strip().splitlines():
        try:
            available.add(int(line))
        except ValueError:
            pass

    bad = sorted(set(gpus) - available)
    if bad:
        raise RuntimeError("GPU IDs " + str(bad) + " not found. Available GPUs: " + str(sorted(available)))


def _check_port_available(port: int):
    """Raise if a port is already in use."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if s.connect_ex(("127.0.0.1", port)) == 0:
        raise RuntimeError("Port " + str(port) + " is already in use")
    s.close()


def _build_extra_args(extra: dict) -> list:
    """Convert a dict of extra vLLM flags to CLI args.

    Underscores in keys become hyphens.  Boolean ``True`` emits a flag
    without a value; other values are stringified.
    """
    args = []
    for key, value in extra.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            args.append(flag)
        elif value is False or value is None:
            pass
        else:
            args.extend([flag, str(value)])
    return args


class VLLMServerManager:
    """Start, health-check, and stop vLLM server subprocesses.

    Usage::

        mgr = VLLMServerManager(
            model="my-model",
            gpus=[0, 1, 2, 3],
            gpus_per_server=2,
        )
        mgr.start()
        try:
            urls = mgr.base_urls  # ["http://127.0.0.1:29500/v1", ...]
            ...  # use the servers
        finally:
            mgr.shutdown()

    Or as a context manager::

        with VLLMServerManager(...) as mgr:
            urls = mgr.base_urls
    """

    def __init__(self, model: str, gpus: list[int], gpus_per_server: int = 1, base_port: int = 29500, extra_args: Optional[dict[str, Any]] = None, log_dir: Optional[Path] = None):
        if not gpus:
            raise ValueError("gpus list must not be empty")
        if gpus_per_server < 1:
            raise ValueError("gpus_per_server must be >= 1")
        if len(gpus) % gpus_per_server != 0:
            raise ValueError("len(gpus)=" + str(len(gpus)) + " is not divisible by gpus_per_server=" + str(gpus_per_server))
        self.model = model
        self.gpus = gpus
        self.gpus_per_server = gpus_per_server
        self.base_port = base_port
        self.extra_args = extra_args or {}
        self.log_dir = log_dir
        self._servers: list[_ServerProcess] = []
        self._started = False

    @property
    def base_urls(self) -> list[str]:
        """Base URLs (with ``/v1`` suffix) for all running servers."""
        return [s.base_url for s in self._servers]

    def start(self):
        """Validate environment, launch servers, and wait until healthy."""
        if self._started:
            raise RuntimeError("Servers already started")
        _check_gpu_ids(self.gpus)
        gpu_groups = self._split_gpus()
        for i, (_, gpu_group) in enumerate(zip(range(len(gpu_groups)), gpu_groups)):
            port = self.base_port + i
            _check_port_available(port)
            self._launch_server(gpu_group, port, server_index=i)
        try:
            self._wait_healthy()
        except Exception:
            self.shutdown()
            raise
        self._started = True
        logger.info("All %d vLLM servers healthy: %s", len(self._servers), self.base_urls)

    def shutdown(self):
        """Stop all server subprocesses."""
        for srv in self._servers:
            self._kill_server(srv)
        self._servers.clear()
        self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def _split_gpus(self) -> list[list[int]]:
        """Partition GPU list into equal-sized groups."""
        n = self.gpus_per_server
        return [self.gpus[i:i + n] for i in range(0, len(self.gpus), n)]

    def _launch_server(self, gpu_group: list[int], port: int, server_index: int = 0):
        cuda_devices = ",".join(str(g) for g in gpu_group)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": cuda_devices}

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model,
            "--port", str(port),
            "--tensor-parallel-size", str(len(gpu_group)),
        ]
        cmd.extend(_build_extra_args(self.extra_args))

        stdout_file = stderr_file = None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = self.log_dir / f"vllm_server_{server_index}.stdout.log"
            stderr_path = self.log_dir / f"vllm_server_{server_index}.stderr.log"
            stdout_file = open(stdout_path, "w")
            stderr_file = open(stderr_path, "w")
            logger.info("Server %d logs: %s, %s", server_index, stdout_path, stderr_path)

        logger.info("Starting vLLM server %d on port %d (GPUs %s): %s", server_index, port, cuda_devices, " ".join(cmd))

        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=stdout_file or subprocess.DEVNULL,
            stderr=stderr_file or subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )
        self._servers.append(_ServerProcess(gpus=gpu_group, port=port, proc=proc, base_url=f"http://127.0.0.1:{port}/v1"))

    def _wait_healthy(self):
        """Poll /v1/models until all servers respond 200 or timeout."""
        deadline = time.monotonic() + _HEALTH_CHECK_TIMEOUT
        remaining = set(range(len(self._servers)))

        while remaining and time.monotonic() < deadline:
            for i in list(remaining):
                srv = self._servers[i]
                ret = srv.proc.poll()
                if ret is not None:
                    raise RuntimeError(
                        "vLLM server " + str(i) + " (port " + str(srv.port) +
                        ", GPUs " + str(srv.gpus) + ") exited with code " + str(ret) + " during startup"
                    )
                url = f"http://127.0.0.1:{srv.port}/v1/models"
                try:
                    req = urllib.request.Request(url, method="GET")
                    urllib.request.urlopen(req, timeout=5)
                    logger.info("Server %d (port %d) is healthy", i, srv.port)
                    remaining.discard(i)
                except (urllib.error.URLError, OSError):
                    pass
            if remaining:
                time.sleep(_HEALTH_CHECK_INTERVAL)

        if remaining:
            ports = [self._servers[i].port for i in remaining]
            raise RuntimeError("vLLM servers on ports " + str(ports) + " did not become healthy within " + str(_HEALTH_CHECK_TIMEOUT) + "s")

    @staticmethod
    def _kill_server(srv: _ServerProcess):
        """Send SIGTERM to the server's process group, then wait."""
        if srv.proc.poll() is not None:
            return
        try:
            pgid = os.getpgid(srv.proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            srv.proc.wait(timeout=15)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(srv.proc.pid), signal.SIGKILL)
                srv.proc.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
        logger.info("Stopped vLLM server on port %d", srv.port)
