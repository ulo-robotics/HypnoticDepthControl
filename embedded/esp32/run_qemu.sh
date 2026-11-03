#!/bin/bash
# $1 target (esp32 / esp32s3 / esp32c3); $2 "build", or QEMU options; $3 the tcp port of the serial (127.0.0.1)
set -e
T=$1
P=$(cd "$(dirname "$0")" && pwd)
. $IDF_PATH/export.sh >/dev/null
if [ "$2" = build ]; then
  cd $P && rm -rf build_$T sdkconfig
  idf.py -B build_$T -D SDKCONFIG=build_$T/sdkconfig set-target $T >/dev/null
  idf.py -B build_$T -D SDKCONFIG=build_$T/sdkconfig build 2>&1 | tail -2
  cd build_$T && esptool.py --chip $T merge_bin --fill-flash-size 4MB -o flash.bin @flash_args >/dev/null
  exit 0
fi
Q=qemu-system-xtensa; [ "$T" = esp32c3 ] && Q=qemu-system-riscv32
cp $P/build_$T/flash.bin /tmp/flash_$3.bin
exec $Q -nographic -machine $T $2 -drive file=/tmp/flash_$3.bin,if=mtd,format=raw -serial tcp:127.0.0.1:$3,server=on,wait=on,nodelay=on -monitor none
