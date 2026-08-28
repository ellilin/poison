#!/usr/bin/env bash
#
# Fetch BackdoorBox and wire the perceptual-hash defense into it.
#
# BackdoorBox is a third-party GPL project and is not vendored into this
# repository. This script clones it at the exact commit the experiments were run
# against, applies four small compatibility fixes (modern PyTorch / NumPy), and
# installs the defense plus its usage example.
#
# Usage:  bash integration/setup.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERE="$ROOT/integration"
DEST="$ROOT/BackdoorBox"
UPSTREAM="https://github.com/THUYimingLi/BackdoorBox.git"
COMMIT="af3afd1c0b172cbf4b0f8c030a9afa0fecc13fc1"

if [ ! -d "$DEST/.git" ]; then
    echo "==> cloning BackdoorBox into $DEST"
    git clone "$UPSTREAM" "$DEST"
fi

echo "==> checking out $COMMIT"
git -C "$DEST" fetch --quiet origin
git -C "$DEST" checkout --quiet "$COMMIT"

echo "==> applying compatibility fixes"
# - core/attacks/TUAP.py          torch.autograd.gradcheck.zero_gradients was
#                                 removed in torch >= 1.9
# - core/attacks/__init__.py      BAAT needs the external ArtFlow repository,
#                                 which is not installable from PyPI
# - core/attacks/Refool.py        np.lib.pad was removed in NumPy 2
# - core/attacks/AdaptivePatch.py the CIFAR-10 branch of the attack was never
#                                 finished upstream (dispatcher commented out,
#                                 leftover single-pattern class, float->PIL bug)
# - core/models/*_curve.py        `import curves` only worked with core/models
#                                 on sys.path
# - core/defenses/__init__.py     register the new PerceptualHash defense
git -C "$DEST" apply --3way "$HERE/compat-fixes.patch"

echo "==> installing the perceptual-hash defense"
cp "$HERE/core/defenses/PerceptualHash.py" "$DEST/core/defenses/PerceptualHash.py"
cp "$HERE/tests/test_PerceptualHash.py" "$DEST/tests/test_PerceptualHash.py"

echo
echo "done. Verify with:"
echo "  python -c \"import sys; sys.path.insert(0,'BackdoorBox'); import core; print(core.PerceptualHash)\""
