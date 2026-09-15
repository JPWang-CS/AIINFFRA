"""CPU state-machine examples for rollout grouping, speculative rollback, and tool waits."""
from dataclasses import dataclass, field


@dataclass
class Sample:
    group_id: str
    sample_id: str
    policy_version: int
    actions: tuple[int, ...]
    old_logp: tuple[float, ...]
    reward: float | None = None
    terminal_reason: str | None = None

    def __post_init__(self):
        if len(self.actions) != len(self.old_logp):
            raise ValueError("one behavior log probability is required per action token")


@dataclass
class RolloutGroup:
    group_id: str
    expected_sample_ids: tuple[str, ...]
    records: dict[str, Sample] = field(default_factory=dict)

    def complete(self, sample):
        if sample.group_id != self.group_id or sample.sample_id not in self.expected_sample_ids:
            raise ValueError("sample does not belong to this prompt group")
        if sample.sample_id in self.records or sample.reward is None or sample.terminal_reason is None:
            raise ValueError("duplicate or incomplete rollout result")
        self.records[sample.sample_id] = sample

    def centered_rewards(self):
        if set(self.records) != set(self.expected_sample_ids):
            raise RuntimeError("the original prompt group is not complete")
        mean = sum(self.records[s].reward for s in self.expected_sample_ids) / len(self.expected_sample_ids)
        return {s: self.records[s].reward - mean for s in self.expected_sample_ids}


def greedy_verify_and_commit(drafts, target_argmax, target_cache_length, bonus_token=None):
    """Greedy illustration only; stochastic sampling needs probability correction."""
    if len(drafts) != len(target_argmax) or target_cache_length < 0:
        raise ValueError("aligned candidate block and nonnegative cache length required")
    accepted = 0
    while accepted < len(drafts) and drafts[accepted] == target_argmax[accepted]:
        accepted += 1
    if accepted < len(drafts):
        emitted = tuple(drafts[:accepted]) + (target_argmax[accepted],)
        return {
            "emitted": emitted,
            "accepted_drafts": accepted,
            "target_cache_length": target_cache_length + accepted,
            "discarded_drafts": tuple(drafts[accepted:]),
            "bonus": None,
        }
    emitted = tuple(drafts) + (() if bonus_token is None else (bonus_token,))
    return {
        "emitted": emitted,
        "accepted_drafts": accepted,
        "target_cache_length": target_cache_length + accepted,
        "discarded_drafts": (),
        "bonus": bonus_token,
    }


@dataclass
class ToolWait:
    kv_policy_version: int
    pending_tools: set[str]
    phase: str = "tool_wait"

    def complete_tool(self, tool_id):
        if self.phase != "tool_wait" or tool_id not in self.pending_tools:
            raise ValueError("unexpected or duplicate tool completion")
        self.pending_tools.remove(tool_id)

    def resume(self, current_policy_version, cache_action):
        if self.pending_tools:
            raise RuntimeError("resume only after all dependent tools are resolved")
        if cache_action not in {"keep", "offload_restore", "recompute"}:
            raise ValueError("unknown KV lifecycle action")
        if cache_action != "recompute" and current_policy_version != self.kv_policy_version:
            raise ValueError("a changed policy version cannot silently reuse exact KV")
        if cache_action == "recompute":
            self.kv_policy_version = current_policy_version
        self.phase = "decode_resume"


def main():
    # Results arrive out of order; group identity, not arrival order, defines comparison.
    group = RolloutGroup("prompt-7", ("s0", "s1", "s2"))
    group.complete(Sample("prompt-7", "s2", 41, (8, 9), (-0.3, -0.5), 1.0, "eos"))
    group.complete(Sample("prompt-7", "s0", 41, (4,), (-0.7,), 0.0, "validator_reject"))
    try:
        group.centered_rewards()
    except RuntimeError:
        pass
    else:
        raise AssertionError("incomplete group must not be normalized")
    group.complete(Sample("prompt-7", "s1", 40, (6, 7), (-0.2, -0.4), 0.5, "tool_return"))
    advantages = group.centered_rewards()
    assert advantages == {"s0": -0.5, "s1": 0.0, "s2": 0.5}
    assert group.records["s1"].old_logp == (-0.2, -0.4)  # never replaced by current-policy logp

    result = greedy_verify_and_commit([11, 12, 13, 14], [11, 12, 99, 1], 40)
    assert result["emitted"] == (11, 12, 99)
    assert result["accepted_drafts"] == 2 and result["target_cache_length"] == 42
    assert result["discarded_drafts"] == (13, 14)

    wait = ToolWait(kv_policy_version=41, pending_tools={"search", "test"})
    wait.complete_tool("search")
    try:
        wait.resume(42, "keep")
    except RuntimeError:
        pass
    else:
        raise AssertionError("must await the second dependent tool")
    wait.complete_tool("test")
    try:
        wait.resume(42, "keep")
    except ValueError:
        pass
    else:
        raise AssertionError("changed weights invalidate exact-cache reuse")
    wait.resume(42, "recompute")
    assert wait.phase == "decode_resume" and wait.kv_policy_version == 42
    print("PASS: out-of-order rollout grouping, old logp identity, candidate commit/rollback, tool barrier, KV version")
    print("Hypothetical CPU state machine; no rollout, model service, network, or GPU trace executed")


if __name__ == "__main__":
    main()
