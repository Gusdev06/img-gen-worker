"""
RunPod Serverless Handler — Geracao de Imagens
===============================================
Modelos suportados: Flux Schnell, Flux Dev
Suporte a LoRA customizada via URL do S3

Exemplo de input:
{
    "input": {
        "prompt": "a cat wearing sunglasses on a beach",
        "model": "flux-schnell",
        "width": 1024,
        "height": 1024,
        "steps": 4,
        "guidance_scale": 0.0,
        "seed": 42,
        "num_images": 1,
        "lora_url": null,
        "lora_scale": 0.8
    }
}

Exemplo de output:
{
    "images": [
        {
            "url": "https://cdn.suaplataforma.com/images/uuid.png",
            "seed": 42,
            "width": 1024,
            "height": 1024
        }
    ],
    "model": "flux-schnell",
    "generation_time": 3.45
}
"""

import os
import time
import torch
import runpod
import requests
import tempfile
from diffusers import FluxPipeline
from utils import validate_input, upload_image_to_s3

# ============================================================
# CARREGAMENTO DE MODELOS (executa 1x na inicializacao do worker)
# ============================================================

print("[INIT] Carregando modelos...")

MODELS = {}

MODEL_REPOS = {
    "flux-schnell": "black-forest-labs/FLUX.1-schnell",
    "flux-dev": "black-forest-labs/FLUX.1-dev",
}

# Carrega o modelo principal (Schnell) na inicializacao
# Os outros sao carregados sob demanda (lazy loading)
PRIMARY_MODEL = os.environ.get("PRIMARY_MODEL", "flux-schnell")

def load_model(model_name: str) -> FluxPipeline:
    """Carrega um modelo do cache local (ja baixado no build)."""
    if model_name in MODELS:
        return MODELS[model_name]

    repo = MODEL_REPOS.get(model_name)
    if not repo:
        raise ValueError(f"Modelo desconhecido: {model_name}")

    print(f"[LOAD] Carregando {model_name}...")
    start = time.time()

    pipe = FluxPipeline.from_pretrained(
        repo,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    # Otimizacoes de memoria
    pipe.enable_model_cpu_offload()

    elapsed = time.time() - start
    print(f"[LOAD] {model_name} carregado em {elapsed:.1f}s")

    MODELS[model_name] = pipe
    return pipe


# Pre-carrega o modelo principal
try:
    load_model(PRIMARY_MODEL)
    print(f"[INIT] Modelo primario '{PRIMARY_MODEL}' pronto!")
except Exception as e:
    print(f"[ERROR] Falha ao carregar modelo primario: {e}")


# ============================================================
# LORA CACHE
# ============================================================

_current_lora_url = None

def apply_lora(pipe: FluxPipeline, lora_url: str, lora_scale: float):
    """
    Baixa e aplica uma LoRA ao pipeline.
    Faz cache — so recarrega se a URL mudar.
    """
    global _current_lora_url

    if lora_url == _current_lora_url:
        # Ja esta carregada, so ajusta a escala
        pipe.fuse_lora(lora_scale=lora_scale)
        return

    # Remove LoRA anterior se houver
    if _current_lora_url is not None:
        try:
            pipe.unfuse_lora()
            pipe.unload_lora_weights()
        except Exception:
            pass

    # Baixa o arquivo .safetensors
    print(f"[LORA] Baixando LoRA: {lora_url}")
    response = requests.get(lora_url, timeout=120)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        f.write(response.content)
        lora_path = f.name

    # Carrega no pipeline
    pipe.load_lora_weights(lora_path)
    pipe.fuse_lora(lora_scale=lora_scale)

    _current_lora_url = lora_url
    print(f"[LORA] LoRA aplicada com sucesso (scale={lora_scale})")

    # Limpa arquivo temporario
    os.unlink(lora_path)


def remove_lora(pipe: FluxPipeline):
    """Remove LoRA do pipeline se houver uma carregada."""
    global _current_lora_url
    if _current_lora_url is not None:
        try:
            pipe.unfuse_lora()
            pipe.unload_lora_weights()
        except Exception:
            pass
        _current_lora_url = None


# ============================================================
# HANDLER PRINCIPAL
# ============================================================

def handler(job: dict) -> dict:
    """
    Handler principal do RunPod Serverless.
    Recebe um job, gera imagem(ns) e retorna URLs.
    """
    try:
        # 1. Valida input
        input_data = job["input"]
        params = validate_input(input_data)

        print(f"[JOB] model={params['model']} | {params['width']}x{params['height']} | "
              f"steps={params['steps']} | images={params['num_images']}")

        # 2. Carrega modelo
        pipe = load_model(params["model"])

        # 3. Aplica ou remove LoRA
        if params["lora_url"]:
            apply_lora(pipe, params["lora_url"], params["lora_scale"])
        else:
            remove_lora(pipe)

        # 4. Configura seed
        generator = None
        seed = params["seed"]
        if seed >= 0:
            generator = torch.Generator("cuda").manual_seed(seed)
        else:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            generator = torch.Generator("cuda").manual_seed(seed)

        # 5. Gera imagens
        start_time = time.time()
        results = []

        for i in range(params["num_images"]):
            current_seed = seed + i
            gen = torch.Generator("cuda").manual_seed(current_seed)

            output = pipe(
                prompt=params["prompt"],
                width=params["width"],
                height=params["height"],
                num_inference_steps=params["steps"],
                guidance_scale=params["guidance_scale"],
                generator=gen,
            )

            image = output.images[0]

            # 6. Upload para S3
            url = upload_image_to_s3(image)

            results.append({
                "url": url,
                "seed": current_seed,
                "width": params["width"],
                "height": params["height"],
            })

            print(f"[JOB] Imagem {i+1}/{params['num_images']} gerada e enviada")

        generation_time = time.time() - start_time

        # 7. Retorna resultado
        return {
            "images": results,
            "model": params["model"],
            "generation_time": round(generation_time, 2),
        }

    except ValueError as e:
        # Erro de validacao (input invalido)
        return {"error": str(e), "type": "validation_error"}

    except torch.cuda.OutOfMemoryError:
        # Sem VRAM suficiente
        torch.cuda.empty_cache()
        return {"error": "GPU sem memoria. Tente resolucao menor.", "type": "oom_error"}

    except Exception as e:
        # Erro generico
        print(f"[ERROR] {type(e).__name__}: {e}")
        return {"error": str(e), "type": "internal_error"}


# ============================================================
# INICIALIZA RUNPOD
# ============================================================

if __name__ == "__main__":
    print("[READY] Worker pronto! Aguardando jobs...")
    runpod.serverless.start({"handler": handler})
