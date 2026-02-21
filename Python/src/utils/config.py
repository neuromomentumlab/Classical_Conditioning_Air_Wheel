import os
import json

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

