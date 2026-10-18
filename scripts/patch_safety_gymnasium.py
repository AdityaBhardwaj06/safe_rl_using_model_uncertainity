"""
Patch Safety-Gymnasium so it imports on Python >= 3.11 (e.g. current Google Colab).

Problem: Safety-Gymnasium declares dataclass fields like `color: np.array = COLOR['hazard']`.
Python 3.11+ rejects unhashable (mutable) dataclass defaults such as numpy arrays, so
`import safety_gymnasium` raises "ValueError: mutable default ... use default_factory".
This rewrites those lines to use `field(default_factory=...)`.

Usage:
    git clone --depth 1 https://github.com/PKU-Alignment/safety-gymnasium.git
    python scripts/patch_safety_gymnasium.py safety-gymnasium
    pip install --no-deps -e safety-gymnasium
    pip install "gymnasium==0.28.1" --no-deps
    pip install cloudpickle farama-notifications jax-jumpy xmltodict pyyaml imageio "mujoco>=3.1,<3.2"
"""
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "safety-gymnasium") / "safety_gymnasium"
pat = re.compile(r"^(\s+\w+: np\.(?:array|ndarray) = )(?!field\()(.+)$")
n_files = n_lines = 0
for py in root.rglob("*.py"):
    src = py.read_text().splitlines(keepends=True)
    changed = False
    for i, line in enumerate(src):
        m = pat.match(line.rstrip("\n"))
        if m:
            src[i] = f"{m.group(1)}field(default_factory=lambda: {m.group(2).strip()})\n"
            changed = True
            n_lines += 1
    if changed:
        text = "".join(src)
        if not re.search(r"^from dataclasses import .*\bfield\b", text, re.M):
            text = re.sub(r"^from dataclasses import (.+)$", r"from dataclasses import \1, field", text, count=1, flags=re.M)
            if "import field" not in text and ", field" not in text:
                text = "from dataclasses import field\n" + text
        py.write_text(text)
        n_files += 1
print(f"Patched {n_lines} dataclass fields in {n_files} files under {root}")
