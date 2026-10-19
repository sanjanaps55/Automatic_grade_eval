"""Pack the finalised model weights into one zip (+ MANIFEST with sizes and SHA-256) for sharing outside git.

Usage: python scripts/package_models.py [--with-resnet-reg] [--all]    -> dist/grade_eval_models.zip
The recipient unzips it into the project root so the files land in models/. Weights are not committed to git.
"""
import hashlib
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL = ["models/marks_sum.pt", "models/hard_b_s0.pt", "models/hard_b_s0.pt.json",   # marks reader: simple rows / hard rows (a|b, sums)
         "models/trocr_reg_small", "models/trocr_name_small"]                          # register number; name (roster hint). Exactly what the app loads
EXTRA = []                                                                              # older models live in models/not_used_models (not packed)
RESNET_REG = []


def files(rel):
    p = ROOT / rel
    return [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file())


def main():
    items = FINAL + (RESNET_REG if "--with-resnet-reg" in sys.argv or "--all" in sys.argv else []) + (EXTRA if "--all" in sys.argv else [])
    out = ROOT / "dist"; out.mkdir(exist_ok=True)
    manifest, zpath = [], out / "grade_eval_models.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in items:
            for f in files(rel):
                name = f.relative_to(ROOT).as_posix()
                manifest.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.stat().st_size:>11}  {name}")
                z.write(f, name)
        z.writestr("MANIFEST.txt", "sha256  bytes  path\n" + "\n".join(manifest) + "\n")
    print(f"{zpath} ({zpath.stat().st_size / 1e6:.0f} MB)"); print("\n".join(manifest))


if __name__ == "__main__":
    main()
