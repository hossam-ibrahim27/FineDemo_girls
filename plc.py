"""PLC link: sends a REJECT pulse to the automation side when a defect is decided.

  * ONE background thread owns the PLC connection (snap7 clients are not thread-safe).
  * The vision code only calls `reject()`, which queues a pulse and returns at once,
    so a slow / dead PLC can never freeze the video stream.
  * Every defect = one separate pulse (ON pulse_ms / OFF gap_ms).
  * During a pulse the command array is [0,0,0,1,0] (element at index 3 = 1),
    otherwise [0,0,0,0,0].
  * Heartbeat toggles every 0.5 s so the PLC can detect a dead PC / cable.
  * If the link drops, queued pulses are DROPPED (a late push would hit a good package).

DB layout (non-optimized):
    Cmd       : Array[0..4] of Bool | Byte | Int   -> offset 0
    Heartbeat : Bool (bit 0), one byte right after the array (even-aligned)
    Running   : Bool (bit 1), same byte as Heartbeat
"""
import struct
import threading
import time

HEARTBEAT_S = 0.5
ARRAY_LEN = 5


# ------------------------------------------------------------------ helpers
def build_array(reject: bool, idx: int):
    """[0,0,0,1,0] while a reject pulse is ON, [0,0,0,0,0] otherwise."""
    return [1 if (reject and i == idx) else 0 for i in range(ARRAY_LEN)]


def array_byte_len(kind: str) -> int:
    return {"bool": 1, "byte": ARRAY_LEN, "int": ARRAY_LEN * 2}[kind]


# ------------------------------------------------------------------ backends
class Snap7Backend:
    def __init__(self, ip, rack=0, slot=1, db=1, port=102, kind="bool", idx=3):
        self.ip, self.rack, self.slot, self.db, self.port = ip, rack, slot, db, port
        self.kind, self.idx = kind, idx
        n = array_byte_len(kind)
        self.flags_offset = n + (n % 2)        # Heartbeat/Running byte, even-aligned
        self.client = None

    def connect(self):
        import snap7
        self.close()
        client = snap7.client.Client()
        client.connect(self.ip, self.rack, self.slot, self.port)
        self.client = client

    def write(self, reject: bool, heartbeat: bool, running: bool):
        arr = build_array(reject, self.idx)
        buf = bytearray(self.flags_offset + 1)
        if self.kind == "bool":
            buf[0] = sum(1 << i for i, v in enumerate(arr) if v)   # bit i = element i
        elif self.kind == "byte":
            buf[0:ARRAY_LEN] = bytes(arr)
        else:  # "int": 16-bit, big-endian
            for i, v in enumerate(arr):
                struct.pack_into(">h", buf, i * 2, v)
        buf[self.flags_offset] = (1 if heartbeat else 0) | (2 if running else 0)
        self.client.db_write(self.db, 0, buf)   # one write = array + flags together

    def close(self):
        if self.client is not None:
            try:
                self.client.disconnect()
            except Exception:
                pass
            self.client = None


class OpcUaBackend:
    """Writes 3 nodes: Cmd array, Heartbeat, Running (in this order)."""

    def __init__(self, url, template, names, kind="bool", idx=3):
        if len(names) != 3:
            raise ValueError("names must hold exactly 3 items: cmd_array,heartbeat,running")
        self.url, self.template, self.names = url, template, names
        self.kind, self.idx = kind, idx
        self.client = None
        self.nodes = []
        self._last = [None] * 3

    def connect(self):
        from asyncua.sync import Client
        self.close()
        client = Client(self.url, timeout=3)
        client.connect()
        self.client = client
        self.nodes = [client.get_node(self.template.format(name=n)) for n in self.names]
        self._last = [None] * 3   # force a full write after (re)connect

    def write(self, reject: bool, heartbeat: bool, running: bool):
        from asyncua import ua
        vt = {"bool": ua.VariantType.Boolean,
              "byte": ua.VariantType.Byte,
              "int": ua.VariantType.Int16}[self.kind]
        arr = build_array(reject, self.idx)
        if self.kind == "bool":
            arr = [bool(v) for v in arr]
        values = [arr, bool(heartbeat), bool(running)]
        types = [vt, ua.VariantType.Boolean, ua.VariantType.Boolean]
        for i, (node, val, t) in enumerate(zip(self.nodes, values, types)):
            if val != self._last[i]:
                node.write_value(ua.DataValue(ua.Variant(val, t)))
                self._last[i] = val

    def close(self):
        if self.client is not None:
            try:
                self.client.disconnect()
            except Exception:
                pass
            self.client = None


# ------------------------------------------------------------------ handler
class PlcHandler:
    def __init__(self, backend=None, pulse_ms=100, gap_ms=60, retry_s=2.0):
        self.backend = backend
        self.enabled = backend is not None
        self.pulse_s = pulse_ms / 1000
        self.gap_s = gap_ms / 1000
        self.retry_s = retry_s

        self._lock = threading.Lock()
        self._pending = 0          # pulses waiting to be sent
        self._running = False
        self._connected = False
        self._last_error = ""
        self._sent_pulses = 0
        self._dropped = 0
        self._stop = threading.Event()
        self._thread = None

    # ---------- called from the vision code (never blocks) ----------
    def reject(self) -> bool:
        """Queue one reject pulse. False if PLC disabled or link is down."""
        if not self.enabled:
            return False
        with self._lock:
            if not self._connected:
                self._dropped += 1
                return False
            self._pending += 1
        return True

    def set_running(self, running: bool):
        with self._lock:
            self._running = bool(running)

    def status(self) -> dict:
        with self._lock:
            return {
                "enabled": self.enabled,
                "connected": self._connected,
                "running": self._running,
                "pending_pulses": self._pending,
                "sent_pulses": self._sent_pulses,
                "dropped_pulses": self._dropped,
                "last_error": self._last_error,
            }

    # ---------- lifecycle ----------
    def start(self):
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="plc-link", daemon=True)
        self._thread.start()

    def close(self, drain_s: float = 1.5):
        """Let queued pulses finish (briefly), then write all zeros and disconnect."""
        if not self.enabled or self._thread is None:
            return
        deadline = time.monotonic() + drain_s
        while time.monotonic() < deadline:
            with self._lock:
                busy = self._pending > 0
            if not busy or not self._connected:
                break
            time.sleep(0.02)
        self._stop.set()
        self._thread.join(timeout=3)
        self._thread = None

    # ---------- worker ----------
    def _link_lost(self, err):
        with self._lock:
            if self._connected:
                print(f"[PLC] link lost: {err} - retrying every {self.retry_s}s")
            self._connected = False
            self._dropped += self._pending
            self._pending = 0
            self._last_error = err
        self.backend.close()

    def _run(self):
        be = self.backend
        heartbeat, last_hb = False, 0.0
        pulse_on, phase_end = False, 0.0     # pulse state machine
        last_flags = None
        next_retry = 0.0

        while not self._stop.is_set():
            now = time.monotonic()

            # ----- (re)connect -----
            if not self._connected:
                if now < next_retry:
                    time.sleep(0.05)
                    continue
                try:
                    be.connect()
                    with self._lock:
                        self._connected = True
                        self._last_error = ""
                    last_flags = None
                    pulse_on = False
                    print("[PLC] connected")
                except Exception as e:
                    self._link_lost(f"{type(e).__name__}: {e}")
                    next_retry = time.monotonic() + self.retry_s
                    continue

            # ----- heartbeat -----
            if now - last_hb >= HEARTBEAT_S:
                heartbeat, last_hb = not heartbeat, now

            # ----- pulse state machine -----
            with self._lock:
                if pulse_on and now >= phase_end:
                    pulse_on, phase_end = False, now + self.gap_s     # OFF gap
                elif not pulse_on and now >= phase_end and self._pending > 0:
                    self._pending -= 1
                    self._sent_pulses += 1
                    pulse_on, phase_end = True, now + self.pulse_s    # ON
                running = self._running

            flags = (pulse_on, heartbeat, running)

            if flags != last_flags:
                try:
                    be.write(*flags)
                    last_flags = flags
                except Exception as e:
                    self._link_lost(f"{type(e).__name__}: {e}")
                    next_retry = time.monotonic() + self.retry_s
                    continue

            time.sleep(0.005)

        # ----- shutdown: everything OFF -----
        if self._connected:
            try:
                be.write(False, False, False)
            except Exception:
                pass
        be.close()
        with self._lock:
            self._connected = False