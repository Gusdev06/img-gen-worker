"""
Download dos modelos durante o build da imagem Docker.
Os modelos ficam salvos no cache do HuggingFace dentro da imagem,
e o RunPod faz snapshot disso pra cold starts rapidos.
"""
import torch
from diffusers import FluxPipeline

MODELS = {
    "flux-schnell": "black-forest-labs/FLUX.1-schnell",
    "flux-dev": "black-forest-labs/FLUX.1-dev",
}

def download():
    for name, repo in MODELS.items():
        print(f"[*] Baixando {name} ({repo})...")
        try:
            FluxPipeline.from_pretrained(
                repo,
                torch_dtype=torch.bfloat16,
            )
            print(f"[OK] {name} baixado com sucesso!")
        except Exception as e:
            print(f"[WARN] Erro ao baixar {name}: {e}")
            print("       Pode ser necessario aceitar a licenca em huggingface.co")

if __name__ == "__main__":
    download()
    print("\n[DONE] Todos os modelos processados!")
