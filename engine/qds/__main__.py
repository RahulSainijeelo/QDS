"""Enable ``python3 -m qds ...`` to drive the command-line front-end."""

from .cli import main

raise SystemExit(main())
