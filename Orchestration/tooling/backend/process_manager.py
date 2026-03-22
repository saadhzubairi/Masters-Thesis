import asyncio
import json
import signal
import subprocess
from typing import Optional


class ProcessManager:
    def __init__(self, max_concurrent: int = 4, on_event=None):
        self.max_concurrent = max_concurrent
        self._processes: dict[str, dict] = {}
        self._on_event = on_event  # callback(run_id, event_dict)

    async def start(self, run_id: str, cmd: list[str], cwd: str = None) -> int:
        running = sum(1 for p in self._processes.values() if p["status"] == "running")
        if running >= self.max_concurrent:
            raise RuntimeError(f"Max concurrent processes ({self.max_concurrent}) reached")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                text=True,
                bufsize=1,
            )
        except OSError as e:
            raise RuntimeError(f"Failed to start process: {e}")

        self._processes[run_id] = {
            "process": proc,
            "pid": proc.pid,
            "status": "running",
            "output_lines": [],
        }
        asyncio.get_event_loop().run_in_executor(None, self._read_output, run_id)
        return proc.pid

    def _read_output(self, run_id: str):
        entry = self._processes.get(run_id)
        if not entry:
            return
        proc = entry["process"]
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                    entry["output_lines"].append(parsed)
                    if self._on_event:
                        self._on_event(run_id, parsed)
                except json.JSONDecodeError:
                    entry["output_lines"].append({"type": "log", "message": line})
            proc.wait()
            if entry["status"] == "running":
                entry["status"] = "complete" if proc.returncode == 0 else "failed"
                if self._on_event:
                    self._on_event(run_id, {"type": "_process_ended", "status": entry["status"]})
        except Exception:
            entry["status"] = "failed"

    def get_status(self, run_id: str) -> Optional[str]:
        entry = self._processes.get(run_id)
        if not entry:
            return None
        return entry["status"]

    async def stop(self, run_id: str):
        entry = self._processes.get(run_id)
        if not entry or entry["status"] != "running":
            return
        proc = entry["process"]
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        entry["status"] = "stopped"

    def get_output_lines(self, run_id: str, since: int = 0) -> list:
        entry = self._processes.get(run_id)
        if not entry:
            return []
        return entry["output_lines"][since:]

    def get_active_count(self) -> int:
        return sum(1 for p in self._processes.values() if p["status"] == "running")
