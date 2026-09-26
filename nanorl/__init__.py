"""nanorl — alignment for tiny models.

Supervised Fine-Tuning (SFT) and Direct Preference Optimisation (DPO), written
from scratch on raw PyTorch. The missing link for tiny local models: pre-training
on its own only completes text; these two losses teach it to answer, and to
prefer the good answer.

The text <-> numbers layer is byte-level (see ``nanorl.tokens``): 256 byte
values plus three specials, nothing to download, nothing that fails on a
character nobody predicted.
"""

from . import tokens, trainer

__version__ = "0.2.0"

__all__ = ["tokens", "trainer", "__version__"]
