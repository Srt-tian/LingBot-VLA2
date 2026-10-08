"""CPU-only regression checks with a controlled SummaryWriter (no ML imports)."""
import importlib.util
import pathlib
import sys
import threading
import time
import types
import unittest


class FakeWriter:
    def __init__(self, **kwargs):
        self.release = threading.Event()
        self.release.set()
        self.closed = False
        self.fail = False

    def add_scalar(self, *args):
        self.release.wait()

    def flush(self):
        self.release.wait()
        if self.fail:
            raise OSError("simulated storage failure")

    def close(self):
        self.closed = True


for name in ("numpy", "torch", "torch.utils", "torch.utils.tensorboard"):
    sys.modules[name] = types.ModuleType(name)
sys.modules["torch.utils.tensorboard"].SummaryWriter = FakeWriter
path = pathlib.Path(__file__).resolve().parents[1] / "lingbotvla/utils/async_tb_writer.py"
spec = importlib.util.spec_from_file_location("bounded_tb", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class WriterTests(unittest.TestCase):
    def test_normal_flush_and_close(self):
        writer = module.AsyncTBWriter("unused")
        writer.add_scalar("loss", 1.0, 1)
        self.assertTrue(writer.flush(timeout=1))
        self.assertTrue(writer.close())
        self.assertTrue(writer._writer.closed)

    def test_stalled_io_returns_and_recovers(self):
        writer = module.AsyncTBWriter("unused")
        writer._writer.release.clear()
        writer.add_scalar("loss", 1.0, 1)
        started = time.monotonic()
        self.assertFalse(writer.flush(timeout=0.05))
        self.assertLess(time.monotonic() - started, 1)
        writer._writer.release.set()
        self.assertTrue(writer.flush(timeout=1))
        self.assertTrue(writer.close())

    def test_storage_exception_not_reported_success(self):
        writer = module.AsyncTBWriter("unused")
        writer._writer.fail = True
        self.assertFalse(writer.flush(timeout=1))
        writer._writer.fail = False
        self.assertTrue(writer.close())


if __name__ == "__main__":
    unittest.main()
