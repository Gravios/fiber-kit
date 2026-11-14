"""FK_SESSION_JOBS / --jobs: _resolve_jobs (the 0 = auto mode and the chunk-count cap)."""
import os
from fiber_kit import fiber_session as fs


def test_explicit_jobs_are_capped_by_the_chunk_count():
    assert fs._resolve_jobs(1, 25) == 1
    assert fs._resolve_jobs(8, 25) == 8
    assert fs._resolve_jobs(128, 25) == 25            # oversubscription buys nothing: chunks are the unit
    assert fs._resolve_jobs(64, 64) == 64


def test_auto_is_one_per_chunk_bounded_by_the_usable_cpus(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(64)), raising=False)
    assert fs._resolve_jobs(0, 25) == 25              # fewer chunks than cores -> the chunks cap it
    assert fs._resolve_jobs(0, 200) == 64             # more chunks than cores -> the cores cap it
    assert fs._resolve_jobs(-1, 200) == 64            # any non-positive value means auto
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1}, raising=False)
    assert fs._resolve_jobs(0, 200) == 2              # a cgroup / taskset quota is respected


def test_never_zero_or_negative():
    assert fs._resolve_jobs(0, 1) == 1
    assert fs._resolve_jobs(0, 0) == 1
    assert fs._resolve_jobs(99, 0) == 1
