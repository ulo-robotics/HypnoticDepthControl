#!/bin/bash
# $1 target, $2 QEMU options (e.g. -icount shift=0), $3 port: the app's bench mode ("B") in QEMU
P=$(cd "$(dirname "$0")" && pwd)
bash $P/run_qemu.sh $1 "$2" $3 >/dev/null 2>&1 &
python3 $P/bench.py $3
kill %1
