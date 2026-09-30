"""Voice changer entry point (module E): reads profiles.json, enters the FreeSimpleGUI event loop.

- Event-handler exceptions always show an sg.popup_error dialog and continue, never exit (see app.gui.App.run).
- With env var BSMOKE=1, only build the window and run one non-blocking tick, skipping the main loop (for smoke tests).
"""

import os
import sys
import traceback

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    try:
        os.chdir(PROJECT_ROOT)
    except OSError:
        pass
    smoke = bool(os.environ.get("BSMOKE"))

    import app.gui as gui

    profiles, profiles_error = gui.load_profiles()
    if profiles_error:
        print("[profiles] %s" % profiles_error, file=sys.stderr)
    return gui.run_app(profiles=profiles, profiles_error=profiles_error, smoke=smoke)


if __name__ == "__main__":
    _smoke = bool(os.environ.get("BSMOKE"))
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        _tb = traceback.format_exc()
        print(_tb, file=sys.stderr)
        if not _smoke:
            try:
                import FreeSimpleGUI as sg

                sg.popup_error("变声器启动失败：\n\n%s" % _tb, title="变声器")
            except Exception:
                pass
        sys.exit(1)
