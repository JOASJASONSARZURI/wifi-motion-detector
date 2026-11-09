import sys

from wifi_motion.cli import main

raise SystemExit(main(["eval", *sys.argv[1:]]))