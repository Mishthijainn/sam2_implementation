
from pathlib import Path

p = Path("checkpoints/sam2.1_hiera_tiny.pt")
if p.exists():
    with open(p, "rb") as f:
        print(f.read(100))
