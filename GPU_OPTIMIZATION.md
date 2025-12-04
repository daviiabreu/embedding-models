# 🚀 Guia de Otimização GPU

Este guia explica como usar GPU para acelerar os modelos de Speech-to-Text (STT) e Text-to-Speech (TTS).

## 📋 Índice

1. [Pré-requisitos](#pré-requisitos)
2. [Whisper STT com GPU](#whisper-stt-com-gpu)
3. [TTS com GPU](#tts-com-gpu)
4. [Otimizações de Performance](#otimizações-de-performance)
5. [Troubleshooting](#troubleshooting)

---

## Pré-requisitos

### Verificar GPU disponível

```bash
# Verificar se CUDA está disponível
python -c "import torch; print('CUDA disponível:', torch.cuda.is_available())"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
```

### Versões instaladas

```bash
# Ver versões do PyTorch
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('CUDA:', torch.version.cuda)"
```

### Requisitos de memória

| Modelo         | Memória GPU Mínima | Recomendado |
| -------------- | ------------------ | ----------- |
| Whisper Tiny   | 1 GB               | 2 GB        |
| Whisper Base   | 1 GB               | 2 GB        |
| Whisper Small  | 2 GB               | 4 GB        |
| Whisper Medium | 5 GB               | 8 GB        |
| Whisper Large  | 10 GB              | 12 GB       |
| Bark Small     | 4 GB               | 6 GB        |
| Coqui XTTS v2  | 6 GB               | 8 GB        |

---

## Whisper STT com GPU

### Uso Básico

O serviço STT agora detecta e usa GPU automaticamente:

```python
from stt.stt_service import transcribe_audio, get_gpu_info

# Verificar GPU
print(get_gpu_info())

# Transcrever (usa GPU automaticamente se disponível)
texto = transcribe_audio("audio.mp3")
```

### Configurações de Performance

```python
# Transcrição com FP16 (2x mais rápido na GPU)
texto = transcribe_audio("audio.mp3", fp16=True)

# Transcrição com FP32 (mais preciso, porém mais lento)
texto = transcribe_audio("audio.mp3", fp16=False)
```

### Testar Performance

```bash
cd stt
python stt_service.py
```

### Modelos disponíveis

```python
# Trocar modelo (edite MODEL_NAME em stt_service.py)
MODEL_NAME = "tiny"    # Mais rápido, menos preciso
MODEL_NAME = "base"    # Rápido
MODEL_NAME = "small"   # Balanceado
MODEL_NAME = "medium"  # Boa qualidade (padrão)
MODEL_NAME = "large"   # Melhor qualidade, mais lento
```

---

## TTS com GPU

### Opção 1: Bark (Recomendado para início rápido)

```python
from tts.tts_service_gpu import initialize_tts_gpu, sentence_to_speech_gpu

# Inicializar com Bark
service = initialize_tts_gpu(model_type="bark")

# Gerar áudio
sentence_to_speech_gpu(
    "Olá, este é um teste.",
    "output.wav",
    speaker="v2/pt_speaker_0"  # Vozes: pt_speaker_0 a pt_speaker_9
)

# Ver uso de GPU
print(service.get_gpu_info())
```

### Opção 2: Coqui TTS (Melhor qualidade)

```bash
# Instalar Coqui TTS
pip install TTS
```

```python
from tts.tts_service_gpu import initialize_tts_gpu, sentence_to_speech_gpu

# Inicializar com Coqui
service = initialize_tts_gpu(model_type="coqui")

# Gerar áudio
sentence_to_speech_gpu("Teste de alta qualidade.", "output.wav")
```

### Exemplos completos

```bash
cd tts
python example_gpu_usage.py
```

---

## Otimizações de Performance

### 1. Half Precision (FP16)

**Benefícios:**

- 2-3x mais rápido
- Usa metade da memória GPU
- Mínima perda de qualidade

**Quando usar:**

- GPUs NVIDIA com Tensor Cores (RTX 20xx+, V100, A100)
- Quando memória é limitada
- Para processamento em lote

**Como ativar:**

```python
# Whisper
texto = transcribe_audio("audio.mp3", fp16=True)

# TTS (ativado automaticamente)
service = initialize_tts_gpu(model_type="bark")
```

### 2. Batch Processing

Para múltiplos arquivos:

```python
import torch
from stt.stt_service import transcribe_audio, clear_gpu_cache

arquivos = ["audio1.mp3", "audio2.mp3", "audio3.mp3"]

for i, arquivo in enumerate(arquivos):
    texto = transcribe_audio(arquivo)
    print(f"{i+1}/{len(arquivos)}: {texto}")

    # Limpar cache periodicamente
    if (i + 1) % 10 == 0:
        clear_gpu_cache()
```

### 3. Otimizações do PyTorch

Já aplicadas automaticamente:

```python
# Benchmark CUDNN
torch.backends.cudnn.benchmark = True

# TF32 para GPUs Ampere+
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### 4. Gerenciamento de Memória

```python
from stt.stt_service import clear_gpu_cache, get_gpu_info

# Ver uso atual
info = get_gpu_info()
print(f"Memória usada: {info['memory_allocated']}")

# Limpar cache
clear_gpu_cache()
```

---

## Benchmarks Esperados

### Whisper Medium (GPU vs CPU)

| Áudio | GPU (RTX 3080) | CPU (i7-10700K) | Speedup |
| ----- | -------------- | --------------- | ------- |
| 30s   | ~2s            | ~10s            | 5x      |
| 1min  | ~3s            | ~20s            | 6.6x    |
| 5min  | ~12s           | ~90s            | 7.5x    |

### Bark TTS (GPU vs CPU)

| Texto       | GPU (RTX 3080) | CPU (i7-10700K) | Speedup |
| ----------- | -------------- | --------------- | ------- |
| 1 frase     | ~2s            | ~8s             | 4x      |
| 1 parágrafo | ~5s            | ~25s            | 5x      |
| Página      | ~15s           | ~90s            | 6x      |

_Nota: Resultados variam conforme hardware_

---

## Troubleshooting

### GPU não detectada

```bash
# Verificar CUDA
nvidia-smi

# Verificar PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# Se False, reinstalar PyTorch com CUDA:
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Out of Memory (OOM)

```python
# Solução 1: Usar modelo menor
MODEL_NAME = "small"  # Em vez de "medium"

# Solução 2: Limpar cache
from stt.stt_service import clear_gpu_cache
clear_gpu_cache()

# Solução 3: Processar em lotes menores
# Processar 1 arquivo por vez e limpar cache
```

### Performance baixa

```bash
# 1. Verificar se está realmente usando GPU
python -c "from stt.stt_service import device; print('Device:', device)"

# 2. Verificar temperatura da GPU
nvidia-smi

# 3. Ativar modo performance (Linux)
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 300  # Ajuste conforme sua GPU
```

### Erros de compatibilidade

```bash
# Verificar versões
pip list | grep -E "torch|cuda|whisper"

# Se necessário, reinstalar:
pip install --upgrade openai-whisper
pip install --upgrade transformers
```

---

## Dicas Adicionais

### 1. Monitorar GPU em tempo real

```bash
# Terminal separado
watch -n 0.5 nvidia-smi
```

### 2. Limitar uso de memória

```python
import torch

# Limitar a 80% da memória GPU
torch.cuda.set_per_process_memory_fraction(0.8)
```

### 3. Multi-GPU

```python
# Especificar GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Usar apenas GPU 0
```

### 4. Processamento assíncrono

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def processar_lote(arquivos):
    with ThreadPoolExecutor(max_workers=2) as executor:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(executor, transcribe_audio, arquivo)
            for arquivo in arquivos
        ]
        return await asyncio.gather(*tasks)

# Usar
arquivos = ["audio1.mp3", "audio2.mp3"]
resultados = asyncio.run(processar_lote(arquivos))
```

---

## Recursos

- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [Whisper GitHub](https://github.com/openai/whisper)
- [Bark GitHub](https://github.com/suno-ai/bark)
- [Coqui TTS](https://github.com/coqui-ai/TTS)

---

## Suporte

Para problemas ou dúvidas:

1. Verifique os logs com `logging.INFO`
2. Execute os testes: `python stt/stt_service.py`
3. Consulte a documentação oficial do PyTorch
