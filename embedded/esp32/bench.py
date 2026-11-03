# connect to the app's UART (QEMU tcp serial), send 'B', print the LETHE lines until LETHE DONE
import socket, sys, time
port = int(sys.argv[1])
for _ in range(100):
    try:
        s = socket.create_connection(("127.0.0.1", port)); break
    except OSError:
        time.sleep(0.2)
buf = b""
while b"LETHE READY" not in buf:
    buf += s.recv(4096)
t0 = time.time()
s.sendall(b"B")
buf = buf[buf.index(b"LETHE READY"):]
while b"LETHE DONE" not in buf:
    buf += s.recv(4096)
for l in buf.decode(errors="replace").splitlines():
    if l.startswith("LETHE"):
        print(l)
print(f"host wall {time.time() - t0:.1f} s")
