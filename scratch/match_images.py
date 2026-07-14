import os
from PIL import Image

src_dir = "frontend ui imgs"
files = [f for f in os.listdir(src_dir) if f.endswith(".png")]

print("Scanning images:")
for f in files:
    path = os.path.join(src_dir, f)
    img = Image.open(path)
    print(f"\nFile: {f}")
    print(f"Size: {img.size}")
    print(f"Metadata: {img.info}")
