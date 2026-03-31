from __future__ import annotations

from pathlib import Path

from config import TRAINED_MODEL, HELD_OUT_TEST_DATA

# is_traing_model_available is used to check whether trained model
# and optional test data file are available or not.

# 1. It checks if trained model file exists.
# 2. If required, it also checks if held-out test data exists.

# If required files are missing, it returns False,
# otherwise it returns True.

def is_traing_model_available(
    require_model: bool = True,
    require_held_out_test_data: bool = False,
) -> bool:
   
    if require_model and not Path(TRAINED_MODEL).exists():
        return False

    if require_held_out_test_data and not Path(HELD_OUT_TEST_DATA).exists():
        return False

    return True