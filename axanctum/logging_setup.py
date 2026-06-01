from __future__ import annotations

import logging
import sys

# ═══════════════════════════════════════════════════════════════════════
# 📋  LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("aksa_v2.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("AksaV2")
