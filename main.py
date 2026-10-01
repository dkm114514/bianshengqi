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

from app.runtime import (AlreadyRunningError, InstanceLock, configure_runtime,
                         install_logging, log_error, migrate_legacy_data)


def main():
    configure_runtime()
    try:
        os.chdir(PROJECT_ROOT)
    except OSError:
        pass
    smoke = bool(os.environ.get("BSMOKE"))

    with InstanceLock():
        install_logging()
        migrate_legacy_data()
        import app.gui as gui

        profiles, profiles_error = gui.load_profiles()
        if profiles_error:
            log_error("[profiles] %s" % profiles_error)
        return gui.run_app(profiles=profiles, profiles_error=profiles_error, smoke=smoke)


if __name__ == "__main__":
    _smoke = bool(os.environ.get("BSMOKE"))
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        _tb = traceback.format_exc()
        log_error(_tb)
        if not _smoke:
            try:
                import FreeSimpleGUI as sg

                message = str(sys.exc_info()[1]) if isinstance(sys.exc_info()[1], AlreadyRunningError) else _tb
                sg.popup_error("变声器启动失败：\n\n%s" % message, title="变声器")
            except Exception:
                pass
        sys.exit(1)
