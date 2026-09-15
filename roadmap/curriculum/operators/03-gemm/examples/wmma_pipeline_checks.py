"""CPU checks of copy ownership and the lesson's two-slot schedule, not GPU execution."""
import unittest


def schedule(rounds, stages):
    if type(rounds) is not int or rounds < 0 or stages not in (1, 2):
        raise ValueError("nonnegative rounds, one or two stages required")
    slots = [None] * stages
    events = []

    def submit(t):
        s = t % stages
        assert slots[s] is None, "overwrite before release"
        slots[s] = (t, "pending")
        events.append(("submit", t, s))

    def ready(t):
        s = t % stages
        assert slots[s] == (t, "pending")
        slots[s] = (t, "ready")
        events.append(("ready", t, s))

    if stages == 2 and rounds:
        submit(0)
        ready(0)
    for t in range(rounds):
        if stages == 1:
            submit(t)
            ready(t)
        elif t + 1 < rounds:
            submit(t + 1)
        s = t % stages
        assert slots[s] == (t, "ready"), "read before copy complete"
        events.append(("compute", t, s))
        slots[s] = None
        events.append(("release", t, s))
        if stages == 2 and t + 1 < rounds:
            ready(t + 1)
    assert all(s is None for s in slots)
    return events


class Checks(unittest.TestCase):
    def test_pairs_cover_one_tile(self):
        writes = []
        for lane in range(32):
            for pair in range(lane, 128, 32):
                r, c = divmod(pair * 2, 16)
                self.assertEqual(c % 2, 0)
                self.assertLess(c + 1, 16)
                writes.extend([(r, c), (r, c + 1)])
        self.assertEqual(len(writes), 256)
        self.assertEqual(set(writes), {(r, c) for r in range(16) for c in range(16)})

    def test_schedule_fill_reuse_drain(self):
        for rounds in (0, 1, 2, 3, 5, 17):
            for stages in (1, 2):
                events = schedule(rounds, stages)
                for action in ("submit", "ready", "compute", "release"):
                    self.assertEqual([t for a, t, s in events if a == action], list(range(rounds)))
                if stages == 2 and rounds > 1:
                    self.assertLess(events.index(("submit", 1, 1)), events.index(("compute", 0, 0)))
                    self.assertLess(events.index(("compute", 0, 0)), events.index(("ready", 1, 1)))

    def test_padded_copy_addresses(self):
        for m, n, k in ((1, 1, 1), (17, 19, 21), (33, 65, 7), (32, 48, 32)):
            mp, np, kp = [((x + 15) // 16) * 16 for x in (m, n, k)]
            for bm in range(0, mp, 16):
                for bk in range(0, kp, 16):
                    for n0 in range(0, np, 16):
                        for pair in range(128):
                            r, c = divmod(pair * 2, 16)
                            a = (bm + r) * np + n0 + c
                            b = (n0 + r) * kp + bk + c
                            self.assertEqual(a * 2 % 4, 0)
                            self.assertEqual(b * 2 % 4, 0)
                            self.assertLess(a + 1, mp * np)
                            self.assertLess(b + 1, np * kp)


if __name__ == "__main__":
    unittest.main()
