import asyncio
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from process_manager import ProcessManager

@pytest.fixture
def manager(tmp_path):
    return ProcessManager(max_concurrent=2)

@pytest.mark.asyncio
async def test_start_process_returns_pid(manager):
    pid = await manager.start("test_1", ["python3", "-c", "import time; print('{\"type\":\"complete\"}'); time.sleep(0.1)"])
    assert isinstance(pid, int)
    assert pid > 0

@pytest.mark.asyncio
async def test_get_status_running(manager):
    pid = await manager.start("test_1", ["python3", "-c", "import time; time.sleep(5)"])
    status = manager.get_status("test_1")
    assert status == "running"
    await manager.stop("test_1")

@pytest.mark.asyncio
async def test_get_status_unknown(manager):
    status = manager.get_status("nonexistent")
    assert status is None

@pytest.mark.asyncio
async def test_stop_kills_process(manager):
    pid = await manager.start("test_1", ["python3", "-c", "import time; time.sleep(60)"])
    await manager.stop("test_1")
    status = manager.get_status("test_1")
    assert status in ("stopped", "failed")

@pytest.mark.asyncio
async def test_max_concurrent_enforced(manager):
    await manager.start("a", ["python3", "-c", "import time; time.sleep(60)"])
    await manager.start("b", ["python3", "-c", "import time; time.sleep(60)"])
    with pytest.raises(RuntimeError, match="concurrent"):
        await manager.start("c", ["python3", "-c", "import time; time.sleep(60)"])
    await manager.stop("a")
    await manager.stop("b")

@pytest.mark.asyncio
async def test_read_output_lines(manager):
    script = "import sys, json; print(json.dumps({'type':'epoch','epoch':1})); sys.stdout.flush()"
    await manager.start("test_1", ["python3", "-c", script])
    await asyncio.sleep(0.5)
    lines = manager.get_output_lines("test_1")
    assert len(lines) >= 1
    assert lines[0]["type"] == "epoch"
