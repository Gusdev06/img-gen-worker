"""
Versão de teste do download_models.py para rodar localmente (CPU/MPS).
Apenas valida que os modelos podem ser baixados, sem precisar de GPU NVIDIA.
"""
import torch
from diffusers import FluxPipeline

MODELS = {
    "flux-schnell": "black-forest-labs/FLUX.1-schnell",
    # "flux-dev": "black-forest-labs/FLUX.1-dev",  # Comentado pra testar só 1
}

def download():
    # Detecta o device disponível
    if torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon
    else:
        device = "cpu"

    print(f"[*] Usando device: {device}")

    for name, repo in MODELS.items():
        print(f"[*] Baixando {name} ({repo})...")
        try:
            # Usa float32 em vez de bfloat16 (melhor compatibilidade CPU)
            pipe = FluxPipeline.from_pretrained(
                repo,
                torch_dtype=torch.float32 if device == "cpu" else torch.float16,
            )
            print(f"[OK] {name} baixado com sucesso!")
            print(f"     Cache salvo em: ~/.cache/huggingface/hub/")
        except Exception as e:
            print(f"[WARN] Erro ao baixar {name}: {e}")
            print("       Pode ser necessario aceitar a licenca em huggingface.co")

if __name__ == "__main__":
    download()
    print("\n[DONE] Teste de download concluído!")
