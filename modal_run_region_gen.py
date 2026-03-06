import modal
import os
import subprocess
import glob

app = modal.App("ultraedit-region-gen")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "accelerate>=0.31.0",
        "clip @ git+https://github.com/openai/CLIP.git",
        "datasets>=2.20.0",
        "huggingface-hub>=0.23.4",
        "numpy>=1.26.4",
        "omegaconf>=2.3.0",
        "openai-clip>=1.0.1",
        "sentencepiece>=0.2.0",
        "tokenizers>=0.19.1",
        "torch>=2.5.0",
        "transformers>=4.41.2",
        "xformers",
        "pillow",
        "open_clip_torch"
    )
)

repo_dir = "/home/nhan/Repo/UltraEdit"
project_mount = modal.Mount.from_local_dir(repo_dir, remote_path="/root/UltraEdit")
hf_cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
ultraedit_cache_volume = modal.Volume.from_name("ultraedit-cache", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100", # Or A10G if preferred
    timeout=3600,
    mounts=[project_mount],
    volumes={
        "/root/.cache/huggingface": hf_cache_volume,
        "/root/.cache/ultraedit": ultraedit_cache_volume
    }
)
def run_job():
    os.chdir("/root/UltraEdit")
    
    cmd = [
        "python", "region_gen_p2p.py",
        "--image", "mask_output/input_resized.png",
        "--mf_mask", "mask_output/Mf_fine_grained.png",
        "--mb_mask", "mask_output/Mb_bounding_box.png",
        "--soft_mask_value", "0.5",
        "--source_caption", "A zoomed-in view of a silver metal fence that has a doorway and is very large. Along the edges of the doorway-sized fence, there are silver cylindrical lines that run horizontally along the top and bottom portions of the fence, to the left and right of the fence, there are two cylindrical poles that run vertically. Along the middle portion of the fence, there is another pole, but this pole runs horizontally. Connecting the poles together are thin, rhombus-shaped wires. Through the fence, a large brick wall that has graffiti along it can be seen. Along the floor, there are tall blades of grass. Along the very bottom portion of the building there are large white markings of graffiti, at the center of each white marking there is black graffiti. Above the words, there is a drawing of the face of a monster, most of which is white. The head of the monster is square, and the eyes of it are made up of two upside-down crosses, while the mouth of it is shaped like a wide u. Sticking out of the top portion of the mouth are two triangular-shaped teeth. It is daytime, as everything can be seen clearly.",
        "--target_caption", "A zoomed-in, clear view of a large, silver chain-link fence and complex gate structure, exactly as described in the source, standing in the foreground, with tall grass and weeds at the bottom and a sidewalk. Through and above the fence, the background is a brown brick wall. The lower portion of the wall retains its faint, original large white markings with black graffiti at their center. Higher up on the wall, the main, large graffiti piece, previously mostly white and known as the monster face or ghost graffiti, is now painted entirely in a solid, deep, vibrant blue. This blue-painted area forms the square-headed monster face. Set into this new blue field are the exact same black detailed features: two black upside-down crosses for eyes, and a black, wide-U-shaped mouth with triangular teeth, all positioned as before and sharply defined against the deep blue background. The fire escape, electrical wires, other wall textures, and the overall hazy daytime lighting remain unchanged.",
        "--use_long_clip",
        "--output_dir", "region_output",
        "--device", "cuda"
    ]
    
    print("Executing command:", " ".join(cmd))
    
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/UltraEdit/diffusers/src:/root/UltraEdit/data_generation:/root/UltraEdit/Long-CLIP"
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
    
    process.wait()
    if process.returncode != 0:
        raise Exception(f"Script failed with return code {process.returncode}")
        
    print("Reading output files...")
    results = {}
    out_files = glob.glob("region_output/*")
    for filepath in out_files:
        with open(filepath, "rb") as f:
            results[filepath] = f.read()
    return results

@app.local_entrypoint()
def main():
    print("Submitting Modal job...")
    results = run_job.remote()
    print(f"Job finished. Received {len(results)} output files.")
    for filepath, content in results.items():
        base_name = os.path.basename(filepath)
        local_path = os.path.join("region_output", base_name)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(content)
        print(f"Saved locally: {local_path}")
