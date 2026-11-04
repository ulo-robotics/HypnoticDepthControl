#!/bin/sh
# build.sh <driver.cpp> <file.c>...: compile the generated C files in parallel (-O3 -march=native; an object file newer than its source is
# kept), then link the driver against IPOPT (pkg-config ipopt). Exits with the first failure.
cd "$(dirname "$0")" || exit 1
s=$(date +%s); t=$1; shift
pids=""
for c in "$@"; do o=${c%.c}.o; if [ ! "$o" -nt "$c" ]; then gcc -O3 -march=native -c "$c" -o "$o" & pids="$pids $!"; fi; done
for p in $pids; do wait "$p" || { echo "build: compiling the generated C failed" >&2; exit 1; }; done
g++ -O3 -march=native -std=c++17 $(pkg-config --cflags ipopt) "$t" $(for c in "$@"; do echo "${c%.c}.o"; done) $(pkg-config --libs ipopt) \
  -o "${t%.cpp}" || exit 1
echo "build ${t%.cpp} in $(( $(date +%s) - s )) s"
