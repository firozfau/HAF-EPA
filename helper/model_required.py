from __future__ import annotations

from pathlib import Path

from config import TRAINED_MODEL, HELD_OUT_TEST_DATA


def is_traing_model_available(
    require_model: bool = True,
    require_held_out_test_data: bool = False,
) -> bool:
   
    if require_model and not Path(TRAINED_MODEL).exists():
        return False

    if require_held_out_test_data and not Path(HELD_OUT_TEST_DATA).exists():
        return False

    return True