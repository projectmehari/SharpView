"""
SharpView · Modal Backend
Deploy:  modal deploy modal_app.py
"""

import modal
from fastapi import FastAPI, Request
from fastapi.responses import Response, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04",
        add_python="3.13",
    )
    .run_commands("apt-get update -qq && apt-get install -y git wget")
    .pip_install("uv")
    .run_commands("uv pip install --system git+https://github.com/apple/ml-sharp.git")
    .run_commands(
        "mkdir -p /root/.cache/torch/hub/checkpoints && "
        "wget -q -O /root/.cache/torch/hub/checkpoints/sharp_2572gikvuh.pt "
        "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
    )
    .pip_install("pillow", "python-multipart", "fastapi", "starlette")
)

app = modal.App("sharpview", image=image)

CHECKPOINT = "/root/.cache/torch/hub/checkpoints/sharp_2572gikvuh.pt"

# Modal Dict for shareable PLY storage (persists across containers)
# Each entry: { "ply": <bytes>, "filename": str, "created": float }
ply_store = modal.Dict.from_name("sharpview-ply-store", create_if_missing=True)

# Custom CORS: echoes back whatever origin the browser sends.
class PermissiveCORS(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        origin = request.headers.get("origin") or "*"
        if request.method == "OPTIONS":
            return Response(
                status_code=204,
                headers={
                    "Access-Control-Allow-Origin": origin,
                    "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
                    "Access-Control-Allow-Headers": "*",
                    "Access-Control-Max-Age": "86400",
                },
            )
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        return response

fastapi_app = FastAPI()
fastapi_app.add_middleware(PermissiveCORS)


@fastapi_app.get("/health")
def health():
    return {"status": "ok", "model": "apple/ml-sharp"}


@fastapi_app.post("/predict")
async def predict(request: Request):
    import tempfile, subprocess, os, pathlib, uuid, time

    content_type = request.headers.get("content-type", "")
    ext = ".jpg"
    filename = "input.jpg"

    try:
        if "multipart" in content_type:
            form = await request.form()
            image_file = form.get("image") or next(iter(form.values()), None)
            if image_file is None:
                return JSONResponse({"error": "no image field"}, status_code=400)
            image_bytes = await image_file.read()
            fname = getattr(image_file, "filename", "input.jpg") or "input.jpg"
            filename = fname
            ext = pathlib.Path(fname).suffix.lower() or ".jpg"
        else:
            image_bytes = await request.body()
            if image_bytes[:4] == b'\x89PNG':
                ext = ".png"
            elif image_bytes[:4] == b'RIFF':
                ext = ".webp"
            else:
                ext = ".jpg"
    except Exception as e:
        return JSONResponse({"error": f"parse error: {e}"}, status_code=400)

    if not image_bytes:
        return JSONResponse({"error": "empty body"}, status_code=400)

    # Also capture depth map from SHARP output
    print(f"Image received: {len(image_bytes)} bytes, ext={ext}")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "in")
        output_dir = os.path.join(tmpdir, "out")
        os.makedirs(input_dir)
        os.makedirs(output_dir)

        input_path = os.path.join(input_dir, f"image{ext}")
        with open(input_path, "wb") as f:
            f.write(image_bytes)

        cmd = ["sharp", "predict", "-i", input_dir, "-o", output_dir, "-c", CHECKPOINT]
        print(f"Running: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        print("stdout:", result.stdout[:400])
        print("stderr:", result.stderr[:400])

        if result.returncode != 0:
            return JSONResponse({
                "error": "SHARP failed",
                "stderr": result.stderr[-500:],
            }, status_code=500)

        ply_files = list(pathlib.Path(output_dir).glob("**/*.ply"))
        if not ply_files:
            all_files = [str(f) for f in pathlib.Path(output_dir).rglob("*")]
            return JSONResponse({"error": "no PLY found", "files": all_files}, status_code=500)

        ply_bytes = ply_files[0].read_bytes()
        print(f"PLY: {len(ply_bytes)//1024}kb")

        # Look for depth map (SHARP outputs depth as PNG/EXR alongside PLY)
        depth_bytes = None
        depth_b64 = None
        for ext_try in ["*.png", "*.jpg", "*.exr"]:
            depth_files = [f for f in pathlib.Path(output_dir).glob(f"**/{ext_try}")
                          if "depth" in f.name.lower() or "disp" in f.name.lower()]
            if depth_files:
                depth_bytes = depth_files[0].read_bytes()
                break

        # If no labeled depth file, grab any PNG output that isn't the input
        if not depth_bytes:
            all_pngs = [f for f in pathlib.Path(output_dir).glob("**/*.png")]
            if all_pngs:
                depth_bytes = all_pngs[0].read_bytes()

        if depth_bytes:
            import base64
            depth_b64 = base64.b64encode(depth_bytes).decode()

        # Store in Modal Dict for shareable links (TTL: 7 days via key prefix)
        share_id = str(uuid.uuid4())[:8]
        try:
            ply_store[share_id] = {
                "ply": ply_bytes,
                "filename": pathlib.Path(filename).stem,
                "created": time.time(),
            }
            print(f"Stored share_id={share_id}")
        except Exception as e:
            print(f"Share store error (non-fatal): {e}")
            share_id = None

        # Return multipart response: ply + depth + share_id as JSON envelope
        import json
        response_data = {
            "share_id": share_id,
            "ply_b64": __import__('base64').b64encode(ply_bytes).decode(),
            "depth_b64": depth_b64,
            "ply_size": len(ply_bytes),
        }

        return JSONResponse(response_data)


@fastapi_app.get("/share/{share_id}")
async def get_share(share_id: str):
    """Return the PLY file for a given share ID."""
    try:
        entry = ply_store[share_id]
    except KeyError:
        return JSONResponse({"error": "share not found or expired"}, status_code=404)

    return Response(
        content=entry["ply"],
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{entry["filename"]}.ply"'},
    )


@fastapi_app.get("/share/{share_id}/info")
async def get_share_info(share_id: str):
    """Return metadata for a share link."""
    import time
    try:
        entry = ply_store[share_id]
    except KeyError:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {
        "share_id": share_id,
        "filename": entry.get("filename"),
        "created": entry.get("created"),
        "ply_size": len(entry["ply"]),
    }


@app.function(
    image=image,
    gpu="A10G",
    timeout=120,
    container_idle_timeout=60,
)
@modal.asgi_app()
def serve():
    return fastapi_app
