import os
import json
from pathlib import Path

# --------------------------------------------------
# FIND CONFIG RELATIVE TO THIS FILE
# --------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
print(PROJECT_ROOT)

CONFIG_PATH = os.path.join(PROJECT_ROOT,"config", "config.json")

# --------------------------------------------------
# LOAD CONFIG
# --------------------------------------------------

with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)

# --------------------------------------------------
# EXPAND USER PATHS
# --------------------------------------------------

CODE_BASE = os.path.expanduser(CONFIG["paths"]["code_base"])
RAW_BASE = os.path.expanduser(CONFIG["paths"]["raw_base"])
PROC_BASE = os.path.expanduser(CONFIG["paths"]["proc_base"])

# --------------------------------------------------
# Load config once
# --------------------------------------------------
def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r") as f:
        cfg = json.load(f)

    # expand ~
    for k, v in cfg["paths"].items():
        cfg["paths"][k] = os.path.expanduser(v)

    return cfg