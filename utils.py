"""
Funcoes utilitarias: upload S3, export de imagens, validacao de input.
"""
import os
import io
import uuid
import boto3
from PIL import Image

# ============================================================
# S3 CLIENT
# ============================================================
# Compativel com: AWS S3, Cloudflare R2, MinIO, Backblaze B2
# Configure via variaveis de ambiente no RunPod

s3_client = None

def get_s3_client():
    global s3_client
    if s3_client is None:
        s3_client = boto3.client(
            "s3",
            endpoint_url=os.environ.get("S3_ENDPOINT"),           # ex: https://xxx.r2.cloudflarestorage.com
            aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),
            aws_secret_access_key=os.environ.get("S3_SECRET_KEY"),
            region_name=os.environ.get("S3_REGION", "auto"),
        )
    return s3_client


def upload_image_to_s3(image: Image.Image, format: str = "png") -> str:
    """
    Faz upload de uma imagem PIL para o S3 e retorna a URL publica.
    """
    client = get_s3_client()
    bucket = os.environ.get("S3_BUCKET", "generations")
    cdn_url = os.environ.get("CDN_URL", "")  # ex: https://cdn.suaplataforma.com

    # Gera nome unico
    filename = f"images/{uuid.uuid4()}.{format}"

    # Converte imagem pra bytes
    buffer = io.BytesIO()
    image.save(buffer, format=format.upper())
    buffer.seek(0)

    # Upload
    content_type = f"image/{format}"
    client.upload_fileobj(
        buffer,
        bucket,
        filename,
        ExtraArgs={"ContentType": content_type},
    )

    # Retorna URL
    if cdn_url:
        return f"{cdn_url}/{filename}"
    else:
        endpoint = os.environ.get("S3_ENDPOINT", "")
        return f"{endpoint}/{bucket}/{filename}"


def upload_video_to_s3(video_path: str) -> str:
    """
    Faz upload de um arquivo de video para o S3 e retorna a URL publica.
    """
    client = get_s3_client()
    bucket = os.environ.get("S3_BUCKET", "generations")
    cdn_url = os.environ.get("CDN_URL", "")

    filename = f"videos/{uuid.uuid4()}.mp4"

    client.upload_file(
        video_path,
        bucket,
        filename,
        ExtraArgs={"ContentType": "video/mp4"},
    )

    if cdn_url:
        return f"{cdn_url}/{filename}"
    else:
        endpoint = os.environ.get("S3_ENDPOINT", "")
        return f"{endpoint}/{bucket}/{filename}"


# ============================================================
# VALIDACAO DE INPUT
# ============================================================

VALID_MODELS = ["flux-schnell", "flux-dev"]

VALID_SIZES = [512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408]

DEFAULT_PARAMS = {
    "flux-schnell": {"steps": 4, "guidance_scale": 0.0},
    "flux-dev": {"steps": 28, "guidance_scale": 3.5},
}


def validate_input(input_data: dict) -> dict:
    """
    Valida e normaliza o input do job.
    Retorna dict limpo ou levanta ValueError.
    """
    # Prompt obrigatorio
    prompt = input_data.get("prompt", "").strip()
    if not prompt:
        raise ValueError("Campo 'prompt' e obrigatorio")
    if len(prompt) > 2000:
        raise ValueError("Prompt muito longo (max 2000 caracteres)")

    # Modelo
    model = input_data.get("model", "flux-schnell")
    if model not in VALID_MODELS:
        raise ValueError(f"Modelo invalido. Use: {VALID_MODELS}")

    # Dimensoes
    width = input_data.get("width", 1024)
    height = input_data.get("height", 1024)
    width = min(max(int(width), 512), 1408)
    height = min(max(int(height), 512), 1408)
    # Arredonda para multiplo de 64 (exigido pelo Flux)
    width = (width // 64) * 64
    height = (height // 64) * 64

    # Steps e guidance
    defaults = DEFAULT_PARAMS.get(model, {})
    steps = input_data.get("steps", defaults.get("steps", 4))
    steps = min(max(int(steps), 1), 50)

    guidance_scale = input_data.get("guidance_scale", defaults.get("guidance_scale", 0.0))
    guidance_scale = min(max(float(guidance_scale), 0.0), 20.0)

    # Seed
    seed = input_data.get("seed", -1)
    seed = int(seed)

    # Numero de imagens
    num_images = input_data.get("num_images", 1)
    num_images = min(max(int(num_images), 1), 4)

    # LoRA (opcional)
    lora_url = input_data.get("lora_url", None)
    lora_scale = input_data.get("lora_scale", 0.8)
    lora_scale = min(max(float(lora_scale), 0.0), 1.5)

    return {
        "prompt": prompt,
        "model": model,
        "width": width,
        "height": height,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "num_images": num_images,
        "lora_url": lora_url,
        "lora_scale": lora_scale,
    }