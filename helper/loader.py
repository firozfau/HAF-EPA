# helper/loader.py

import sys
import threading
import time

_loading = False
_loader_thread = None


def _animate(message="Working..."):
    dots = 0

    while _loading:
        dots = (dots + 1) % 4
        sys.stdout.write(f"\r{message}{'.' * dots}{' ' * (3 - dots)}")
        sys.stdout.flush()
        time.sleep(0.5)


def show_loader(message="Working..."):
    global _loading, _loader_thread

    _loading = True
    _loader_thread = threading.Thread(
        target=_animate,
        args=(message,),
        daemon=True
    )
    _loader_thread.start()


def hide_loader():
    global _loading

    _loading = False

    if _loader_thread:
        _loader_thread.join()

    sys.stdout.write("\r" + " " * 100 + "\r")
    sys.stdout.flush()