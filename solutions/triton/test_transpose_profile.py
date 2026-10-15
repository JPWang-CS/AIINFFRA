import argparse
import importlib.util
import pathlib
import unittest
from unittest import mock


MODULE = pathlib.Path(__file__).with_name("transpose_profile.py")
SPEC = importlib.util.spec_from_file_location("transpose_profile", MODULE)
profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile)


class FakeCUDA:
    def __init__(self, fail_sync=False):
        self.events = []
        self.fail_sync = fail_sync
        self.sync_calls = 0
        self.nvtx = self
        self.profiler = self

    def start(self): self.events.append("start")
    def stop(self): self.events.append("stop")
    def range_push(self, label): self.events.append(("push", label))
    def range_pop(self): self.events.append("pop")
    def synchronize(self):
        self.events.append("sync")
        self.sync_calls += 1
        if self.fail_sync and self.sync_calls > 1:
            raise RuntimeError("sync failed")


class ProfileTests(unittest.TestCase):
    def test_help_has_no_gpu_dependencies(self):
        with mock.patch.dict("sys.modules", {"torch": None, "triton": None}):
            with self.assertRaises(SystemExit) as raised:
                profile.main(["--help"])
        self.assertEqual(raised.exception.code, 0)

    def test_args_and_positive_validation(self):
        args = profile.parse_args([])
        self.assertEqual((args.rows, args.cols, args.warmup, args.repeats, args.impl), (4096, 4096, 20, 100, "both"))
        for value in ("0", "-1", "x"):
            with self.assertRaises(argparse.ArgumentTypeError):
                profile.positive_int(value)

    def test_capture_order_count_and_sync(self):
        torch = type("Torch", (), {"cuda": FakeCUDA()})()
        calls = []
        profile.run_capture(torch, [("transpose/triton", lambda: calls.append("t")), ("transpose/pytorch", lambda: calls.append("p"))], 3)
        self.assertEqual(calls, ["t"] * 3 + ["p"] * 3)
        self.assertEqual(torch.cuda.events.count("sync"), 3)
        self.assertEqual(torch.cuda.events[0], "sync")
        self.assertNotIn("start", torch.cuda.events)
        self.assertNotIn("stop", torch.cuda.events)
        self.assertEqual(torch.cuda.events.count("pop"), 2)

    def test_capture_api_is_optional_and_brackets_ranges(self):
        cuda = FakeCUDA()
        torch = type("Torch", (), {"cuda": cuda})()
        profile.run_capture(torch, [("transpose/triton", lambda: None)], 1, True)
        self.assertEqual(cuda.events[:2], ["sync", "start"])
        self.assertEqual(cuda.events[-1], "stop")

    def test_capture_cleanup_on_operation_or_sync_failure(self):
        for fail_sync in (False, True):
            cuda = FakeCUDA(fail_sync=fail_sync)
            torch = type("Torch", (), {"cuda": cuda})()
            operation = mock.Mock(side_effect=RuntimeError("operation failed")) if not fail_sync else mock.Mock()
            with self.assertRaises(RuntimeError):
                profile.run_capture(torch, [("transpose/triton", operation)], 1, True)
            self.assertEqual(cuda.events[0:2], ["sync", "start"])
            self.assertEqual(cuda.events[-2:], ["pop", "stop"])


if __name__ == "__main__":
    unittest.main()
