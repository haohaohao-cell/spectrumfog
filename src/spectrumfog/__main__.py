"""``python -m spectrumfog`` 的入口，等价于 ``spectrumfog`` 命令。"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
