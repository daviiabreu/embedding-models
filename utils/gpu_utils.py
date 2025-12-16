"""
Módulo de utilitários compartilhados para serviços de ML com GPU
Fornece funções reutilizáveis para detecção de GPU, gerenciamento de memória e I/O
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import torch

# ==================== DETECÇÃO E CONFIGURAÇÃO DE HARDWARE ====================


def setup_device(enable_optimizations: bool = True) -> str:
    """
    Configura e detecta o dispositivo disponível (GPU ou CPU).

    Args:
        enable_optimizations: Se True, ativa otimizações CUDA

    Returns:
        str: 'cuda' se GPU disponível, caso contrário 'cpu'
    """
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logging.info(f"GPU disponível: {gpu_name}")
        logging.info(f"Memória GPU: {gpu_memory:.2f} GB")

        if enable_optimizations:
            # Otimizações para acelerar operações na GPU
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logging.info("Otimizações CUDA ativadas")
    else:
        device = "cpu"
        logging.warning("GPU não disponível, usando CPU")

    return device


def get_gpu_info(device: str) -> Dict[str, any]:
    """
    Retorna informações detalhadas sobre o uso de GPU.

    Args:
        device: Dispositivo atual ('cuda' ou 'cpu')

    Returns:
        dict: Informações sobre GPU ou indicação de uso de CPU
    """
    if device != "cuda":
        return {"device": "cpu", "available": False}

    return {
        "device": "cuda",
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1e9:.2f} GB",
        "memory_reserved": f"{torch.cuda.memory_reserved(0) / 1e9:.2f} GB",
        "memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB",
        "memory_free": f"{(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1e9:.2f} GB",
    }


def clear_gpu_cache(device: str):
    """
    Limpa cache da GPU para liberar memória.

    Args:
        device: Dispositivo atual ('cuda' ou 'cpu')
    """
    if device == "cuda":
        torch.cuda.empty_cache()
        logging.info("Cache da GPU limpo")
        free_memory = (
            torch.cuda.get_device_properties(0).total_memory
            - torch.cuda.memory_reserved(0)
        ) / 1e9
        logging.info(f"Memória GPU livre: {free_memory:.2f} GB")


def log_gpu_memory(device: str, label: str = ""):
    """
    Registra o uso atual de memória GPU nos logs.

    Args:
        device: Dispositivo atual ('cuda' ou 'cpu')
        label: Label opcional para identificar o momento
    """
    if device == "cuda":
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        prefix = f"[{label}] " if label else ""
        logging.info(
            f"{prefix}GPU: {allocated:.2f} GB alocado, {reserved:.2f} GB reservado"
        )


def set_gpu_memory_fraction(fraction: float = 0.9):
    """
    Limita o uso de memória GPU a uma fração específica.

    Args:
        fraction: Fração da memória total (0.0 a 1.0)
    """
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)
        logging.info(f"Memória GPU limitada a {fraction * 100:.0f}%")


# ==================== GERENCIAMENTO DE DIRETÓRIOS E ARQUIVOS ====================


def setup_output_directory(
    base_dir: Optional[Path] = None, subdir: str = "output_audio"
) -> Path:
    """
    Cria o diretório de saída padrão.

    Args:
        base_dir: Diretório base (se None, usa diretório do módulo chamador)
        subdir: Nome do subdiretório a criar

    Returns:
        Path: Caminho do diretório de saída
    """
    if base_dir is None:
        # Usa o diretório pai do arquivo que chamou esta função
        import inspect

        caller_file = inspect.stack()[1].filename
        base_dir = Path(caller_file).parent.parent

    output_dir = base_dir / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Diretório de saída: {output_dir}")
    return output_dir


def normalize_output_path(
    output_path: str,
    default_dir: Path,
    extension: str = ".wav",
    valid_extensions: Optional[list] = None,
) -> Path:
    """
    Normaliza o caminho de saída do arquivo de áudio.

    Args:
        output_path: Caminho de saída fornecido
        default_dir: Diretório padrão se caminho não for absoluto
        extension: Extensão padrão do arquivo
        valid_extensions: Lista de extensões válidas (se None, usa ['.wav', '.mp3'])

    Returns:
        Path: Caminho normalizado
    """
    if valid_extensions is None:
        valid_extensions = [".wav", ".mp3"]

    output_path = Path(output_path)

    # Se não é absoluto, usar diretório padrão
    if not output_path.is_absolute():
        output_path = default_dir / output_path.name

    # Garantir extensão correta
    if output_path.suffix.lower() not in valid_extensions:
        output_path = output_path.with_suffix(extension)

    # Criar diretório se não existir
    output_path.parent.mkdir(parents=True, exist_ok=True)

    return output_path


def validate_file_exists(filepath: str, file_type: str = "arquivo") -> bool:
    """
    Valida se um arquivo existe e registra erro se não existir.

    Args:
        filepath: Caminho do arquivo
        file_type: Tipo do arquivo para mensagem de erro

    Returns:
        bool: True se existe, False caso contrário
    """
    if not Path(filepath).exists():
        logging.error(f"{file_type.capitalize()} não encontrado: {filepath}")
        return False
    return True


def validate_text_input(text: Optional[str]) -> bool:
    """
    Valida se o texto de entrada é válido.

    Args:
        text: Texto a validar

    Returns:
        bool: True se válido, False caso contrário
    """
    if not text or not text.strip():
        logging.error("Texto vazio fornecido")
        return False
    return True


# ==================== CONVERSÕES E FORMATAÇÃO ====================


def format_file_size(size_bytes: int) -> str:
    """
    Formata tamanho de arquivo para formato legível.

    Args:
        size_bytes: Tamanho em bytes

    Returns:
        str: Tamanho formatado (KB, MB, etc)
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def format_time(seconds: float) -> str:
    """
    Formata tempo em segundos para formato legível.

    Args:
        seconds: Tempo em segundos

    Returns:
        str: Tempo formatado
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


# ==================== BENCHMARK E PERFORMANCE ====================


class PerformanceMonitor:
    """Monitor de performance para operações de ML"""

    def __init__(self, device: str):
        self.device = device
        self.start_time = None
        self.start_memory = None

    def start(self):
        """Inicia monitoramento"""
        import time

        self.start_time = time.time()
        if self.device == "cuda":
            torch.cuda.synchronize()
            self.start_memory = torch.cuda.memory_allocated(0)

    def stop(self) -> Dict[str, any]:
        """
        Para monitoramento e retorna estatísticas

        Returns:
            dict: Estatísticas de performance
        """
        import time

        if self.start_time is None:
            return {}

        elapsed = time.time() - self.start_time

        stats = {
            "elapsed_time": elapsed,
            "elapsed_time_formatted": format_time(elapsed),
        }

        if self.device == "cuda":
            torch.cuda.synchronize()
            end_memory = torch.cuda.memory_allocated(0)
            memory_used = (end_memory - self.start_memory) / 1e9
            stats.update(
                {
                    "gpu_memory_used": f"{memory_used:.2f} GB",
                    "gpu_memory_current": f"{end_memory / 1e9:.2f} GB",
                }
            )

        return stats


# ==================== INFORMAÇÕES DO SISTEMA ====================


def print_system_info(device: str):
    """
    Imprime informações detalhadas do sistema.

    Args:
        device: Dispositivo atual
    """
    print("NFORMAÇÕES DO SISTEMA")

    print(f"\nPyTorch: {torch.__version__}")
    print(f"Dispositivo: {device.upper()}")

    if device == "cuda":
        print("\nGPU:")
        print(f"   Nome: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA: {torch.version.cuda}")

        props = torch.cuda.get_device_properties(0)
        print(f"   Memória total: {props.total_memory / 1e9:.2f} GB")
        print(f"   Multiprocessadores: {props.multi_processor_count}")
        print(f"   Compute Capability: {props.major}.{props.minor}")

        # Otimizações ativas
        print("\n⚡ Otimizações:")
        print(f"   CUDNN Benchmark: {torch.backends.cudnn.benchmark}")
        print(f"   TF32 (matmul): {torch.backends.cuda.matmul.allow_tf32}")
        print(f"   TF32 (cudnn): {torch.backends.cudnn.allow_tf32}")
