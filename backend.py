"""
Módulo de gerenciamento de backends de execução (CPU e CUDA GPU).

Permite seleção explícita ou automática de backends sem exigir a presença
de ambiente CUDA ou GPU NVIDIA para a execução em CPU.
"""

from __future__ import annotations
import os
import sys
import time
from typing import Literal, Any
import numpy as np

BackendName = Literal["cpu", "cuda", "auto"]


class CudaNotAvailableError(RuntimeError):
    """Exceção levantada quando o backend CUDA é requisitado mas não está disponível."""
    pass


def _check_cuda_availability() -> tuple[bool, str]:
    """
    Verifica se CUDA e CuPy estão disponíveis e funcionais.
    Retorna uma tupla (disponível: bool, motivo: str).
    """
    try:
        import cupy as cp
    except ImportError:
        return False, "O pacote 'cupy' não está instalado no ambiente Python."

    try:
        num_devices = cp.cuda.runtime.getDeviceCount()
        if num_devices <= 0:
            return False, "Nenhum dispositivo NVIDIA CUDA foi detectado pelo driver."
        # Tenta inicializar o contexto do dispositivo 0
        with cp.cuda.Device(0):
            _ = cp.zeros(1)
        return True, f"{num_devices} dispositivo(s) CUDA disponível(is)."
    except Exception as exc:
        return False, f"Falha ao inicializar o runtime CUDA: {exc}"


# Cache da verificação para evitar overhead de chamadas repetidas ao driver
_CUDA_AVAILABLE, _CUDA_REASON = _check_cuda_availability()


def is_cuda_available() -> bool:
    """Retorna True se o backend CUDA estiver disponível e pronto para uso."""
    global _CUDA_AVAILABLE, _CUDA_REASON
    return _CUDA_AVAILABLE


def get_cuda_status() -> tuple[bool, str]:
    """Retorna o estado de disponibilidade do CUDA e a mensagem descritiva."""
    global _CUDA_AVAILABLE, _CUDA_REASON
    return _CUDA_AVAILABLE, _CUDA_REASON


def get_backend_module(backend: BackendName = "auto") -> Any:
    """
    Retorna o módulo de array apropriado (numpy ou cupy).

    Parâmetros
    ----------
    backend : {'cpu', 'cuda', 'auto'}
        Nome do backend desejado.

    Retorna
    -------
    module
        numpy para CPU ou cupy para CUDA.

    Lança
    -----
    CudaNotAvailableError
        Se 'cuda' for explicitamente solicitado mas não estiver disponível.
    ValueError
        Se o nome do backend for desconhecido.
    """
    if backend == "auto":
        backend = "cuda" if is_cuda_available() else "cpu"

    if backend == "cpu":
        return np

    if backend == "cuda":
        if not is_cuda_available():
            _, reason = get_cuda_status()
            raise CudaNotAvailableError(
                f"O backend CUDA foi solicitado, mas não está disponível neste ambiente.\n"
                f"Motivo: {reason}\n"
                f"Dica: utilize backend='cpu' ou execute em um ambiente com GPU NVIDIA e CuPy configurado."
            )
        import cupy as cp
        return cp

    raise ValueError(f"Backend desconhecido: '{backend}'. Opções válidas: 'cpu', 'cuda', 'auto'.")


def resolve_backend(backend: BackendName = "auto") -> Literal["cpu", "cuda"]:
    """Resolve o backend para 'cpu' ou 'cuda' concreto."""
    if backend == "auto":
        return "cuda" if is_cuda_available() else "cpu"
    if backend == "cuda" and not is_cuda_available():
        _, reason = get_cuda_status()
        raise CudaNotAvailableError(
            f"O backend CUDA foi solicitado, mas não está disponível neste ambiente.\n"
            f"Motivo: {reason}\n"
            f"Dica: utilize backend='cpu' ou execute em um ambiente com GPU NVIDIA e CuPy configurado."
        )
    if backend not in ("cpu", "cuda"):
        raise ValueError(f"Backend desconhecido: '{backend}'. Opções válidas: 'cpu', 'cuda', 'auto'.")
    return backend


def to_cpu(arr: Any) -> np.ndarray:
    """
    Converte qualquer array (NumPy ou CuPy) para um np.ndarray na CPU.
    Não realiza cópia desnecessária se o array já for um ndarray na CPU.
    """
    if arr is None:
        return None
    # Verifica se é um array CuPy
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def to_device(arr: Any, backend: BackendName = "auto") -> Any:
    """
    Converte um array para o dispositivo correspondente ao backend.
    Se o backend for 'cpu', retorna np.asarray(arr).
    Se o backend for 'cuda', retorna cp.asarray(arr).
    """
    xp = get_backend_module(backend)
    if arr is None:
        return None
    return xp.asarray(arr)


def synchronize(backend: BackendName = "auto") -> None:
    """
    Garante sincronização da execução se o backend for CUDA.
    No backend CPU, é uma operação sem custo (no-op).
    """
    b = resolve_backend(backend)
    if b == "cuda":
        import cupy as cp
        cp.cuda.Stream.null.synchronize()


class DeviceTimer:
    """
    Cronômetro de alta precisão compatível com CPU e CUDA.
    Em CUDA, utiliza cuda.Event para medição precisa no device, evitando
    distorções por lançamentos assíncronos de kernel.
    """

    def __init__(self, backend: BackendName = "auto"):
        self.backend = resolve_backend(backend)
        self.elapsed_time: float = 0.0
        self._start_cpu: float = 0.0
        self._start_event = None
        self._end_event = None

        if self.backend == "cuda":
            import cupy as cp
            self._start_event = cp.cuda.Event()
            self._end_event = cp.cuda.Event()

    def __enter__(self) -> DeviceTimer:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def start(self) -> None:
        if self.backend == "cuda":
            self._start_event.record()
        else:
            self._start_cpu = time.perf_counter()

    def stop(self) -> float:
        if self.backend == "cuda":
            self._end_event.record()
            self._end_event.synchronize()
            # CuPy retorna tempo em milissegundos; convertemos para segundos
            self.elapsed_time = cp_time = self._start_event.elapsed_time(self._end_event) / 1000.0
        else:
            self.elapsed_time = time.perf_counter() - self._start_cpu
        return self.elapsed_time


def get_gpu_memory_info() -> dict[str, int] | None:
    """
    Retorna estatísticas da memória VRAM da GPU em bytes (usada, total, livre),
    ou None caso o backend CUDA não esteja disponível.
    """
    if not is_cuda_available():
        return None
    try:
        import cupy as cp
        dev = cp.cuda.Device()
        free_bytes, total_bytes = dev.mem_info
        used_bytes = total_bytes - free_bytes
        return {
            "free_bytes": free_bytes,
            "total_bytes": total_bytes,
            "used_bytes": used_bytes,
        }
    except Exception:
        return None

