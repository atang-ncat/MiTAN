#!/usr/bin/env python3
"""
Standalone Arduino serial test — NO ROS, NO Docker.
Run directly on the host to test motor direction.

Usage:
    python3 test_serial.py
    python3 test_serial.py /dev/ttyACM1   # if different port
"""

import sys
import time
import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyACM0'
BAUD = 115200

print(f"\n{'='*60}")
print(f"  STANDALONE SERIAL TEST — {PORT}")
print(f"{'='*60}\n")

# ── Open port ────────────────────────────────────────────────
print(f"[1] Opening {PORT} at {BAUD} baud...")
try:
    ser = serial.Serial()
    ser.port = PORT
    ser.baudrate = BAUD
    ser.timeout = 0.5
    ser.dtr = False
    ser.rts = False
    ser.open()
    print(f"    ✓ Port opened (DTR={ser.dtr}, RTS={ser.rts})")
except Exception as e:
    print(f"    ✗ FAILED: {e}")
    print(f"    Make sure Arduino IDE is closed!")
    print(f"    Run: fuser {PORT}")
    sys.exit(1)

# ── Wait for Arduino boot (it may reset) ─────────────────────
print(f"\n[2] Waiting 3 seconds for Arduino boot...")
time.sleep(3)

# ── Drain and print any Arduino boot output ──────────────────
print(f"\n[3] Reading Arduino output:")
boot_lines = []
while ser.in_waiting:
    line = ser.readline().decode('ascii', errors='replace').strip()
    if line:
        print(f"    ARDUINO> {line}")
        boot_lines.append(line)

if not boot_lines:
    print("    (no output received)")
    print("    Note: Arduino may have already completed boot before we opened port")

# ── Send test commands ────────────────────────────────────────
def send_cmd(cmd_str, description, duration=3):
    """Send a command and read Arduino responses."""
    print(f"\n{'─'*60}")
    print(f"  TEST: {description}")
    print(f"  Sending: {cmd_str.strip()}")
    print(f"  Duration: {duration}s")
    print(f"{'─'*60}")

    start = time.time()
    send_count = 0

    while time.time() - start < duration:
        ser.write(cmd_str.encode('ascii'))
        send_count += 1

        # Read any Arduino responses
        while ser.in_waiting:
            line = ser.readline().decode('ascii', errors='replace').strip()
            if line:
                print(f"    ARDUINO> {line}")

        time.sleep(0.05)  # 20 Hz command rate

    print(f"    Sent {send_count} commands in {duration}s")

    # Final read
    time.sleep(0.1)
    while ser.in_waiting:
        line = ser.readline().decode('ascii', errors='replace').strip()
        if line:
            print(f"    ARDUINO> {line}")


# ── STOP first ───────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  MOTOR DIRECTION TEST")
print(f"  Watch your wheels carefully!")
print(f"{'='*60}")

send_cmd("S0,A92\n", "STOP (baseline)", duration=2)

input("\n>>> Press ENTER to test FORWARD (S44) for 3 seconds...")
send_cmd("S44,A92\n", "FORWARD: S44 (speed=44, steering=center)", duration=3)

send_cmd("S0,A92\n", "STOP", duration=2)

input("\n>>> Press ENTER to test BACKWARD (S-44) for 3 seconds...")
send_cmd("S-44,A92\n", "BACKWARD: S-44 (speed=-44, steering=center)", duration=3)

send_cmd("S0,A92\n", "STOP", duration=2)

# ── Summary ──────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  TEST COMPLETE")
print(f"{'='*60}")
print(f"""
  QUESTIONS:
  1. Did S44 (forward) make the wheels go FORWARD or BACKWARD?
  2. Did S-44 (backward) make the wheels go the OPPOSITE direction?
  3. Were both directions the SAME?

  If S44 went backward → set FORWARD_DIR=LOW in the .ino and re-upload
  If both went the same direction → wiring or motor driver issue
""")

# Cleanup
ser.write(b'S0,A92\n')
ser.close()
print("Port closed.\n")
