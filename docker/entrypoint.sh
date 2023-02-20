#! /usr/bin/env bash
set -e -x

# launch paraview
/opt/paraview/bin/pvpython          \
    /opt/vizer/server.py --server   \
    --venv /opt/trame/env           \
    -i 0.0.0.0                      \
    -p 8080                         \
    "$@"
