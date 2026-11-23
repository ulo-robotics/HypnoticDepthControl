#!/bin/bash
# $1 target, $2 first port, $3 instances: QEMU listening on 0.0.0.0:port (one per instance) until killed (for hil.py)
P=$(cd "$(dirname "$0")" && pwd)
. $IDF_PATH/export.sh >/dev/null
Q=qemu-system-xtensa; [ "$1" = esp32c3 ] && Q=qemu-system-riscv32
for i in $(seq 0 $(($3 - 1))); do
  p=$(($2 + i)); cp $P/build_$1/flash.bin /tmp/f$p.bin
  $Q -nographic -machine $1 -drive file=/tmp/f$p.bin,if=mtd,format=raw -serial tcp:0.0.0.0:$p,server=on,wait=on,nodelay=on -monitor none >/dev/null 2>&1 &
done
wait
