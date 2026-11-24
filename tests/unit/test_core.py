"""Contract tests for the eight public DRBG state-machine operations."""

import pytest

from drbg.core import (
    check_health,
    deserialize,
    drbg_batch,
    generate,
    instantiate,
    predict_resistance,
    reseed,
    serialize,
)


def test_reseed_combines_old_state_with_new_entropy():
    """Defect 1: reseeding must derive from old state plus new entropy."""
    initial = bytes([0x11]) * 32
    new_entropy = bytes([0x33]) * 32
    base = instantiate(initial, b"reseed-nonce", b"reseed-mix")
    advanced = generate(base, 32, b"advance-one")["state"]
    other = instantiate(initial, b"reseed-nonce", b"reseed-mix")

    from_advanced = reseed(advanced, new_entropy, b"reseed-additional")
    from_fresh = reseed(other, new_entropy, b"reseed-additional")

    assert from_advanced["entropy_count"] == 64
    assert from_advanced["key"] != from_fresh["key"]
    assert (
        generate(from_advanced, 32, b"after-reseed")["output"]
        != generate(from_fresh, 32, b"after-reseed")["output"]
    )


def test_generate_advances_state_and_changes_output():
    """Defect 2: generation must advance the block counter and key."""
    state = instantiate(bytes([0x21]) * 32, b"advance-nonce", b"advance")
    first = generate(state, 32, b"sequence")
    second = generate(first["state"], 32, b"sequence")

    assert len(first["output"]) == 32
    assert first["output"] != second["output"]
    assert first["state"]["generation"] == 1
    assert second["state"]["generation"] == 2
    assert second["state"]["counter"] > first["state"]["counter"]


def test_serialize_writes_version_header():
    """Defect 3: the blob must begin with magic and the format version."""
    state = instantiate(bytes([0x31]) * 32, b"version-nonce", b"version")
    blob = serialize(state)

    assert blob[:4] == b"DRBG"
    assert blob[4] == 1
    assert len(blob) >= 5 + 4 + 32


def test_deserialize_rejects_flipped_checksum():
    """Defect 4: a single flipped bit in the checksum must be rejected."""
    state = instantiate(bytes([0x41]) * 32, b"checksum-nonce", b"checksum")
    blob = bytearray(serialize(state))
    blob[-1] ^= 0x01

    with pytest.raises(ValueError):
        deserialize(bytes(blob))


def test_prediction_resistance_invalidates_old_state():
    """Defect 5: a PR refresh must make the old state unusable."""
    state = instantiate(bytes([0x51]) * 32, b"pr-nonce", b"pr-invalidate")
    refreshed = predict_resistance(state, b"pr-additional")

    with pytest.raises(ValueError):
        generate(state, 16, b"old-state")
    assert len(generate(refreshed, 16, b"new-state")["output"]) == 16


def test_instantiate_rejects_insufficient_entropy():
    """Defect 6: short entropy must raise, not be zero-padded."""
    with pytest.raises(ValueError):
        instantiate(b"", b"short-nonce", b"short-entropy")
    with pytest.raises(ValueError):
        instantiate(bytes([0x61]) * 31, b"short-nonce", b"short-entropy")


def test_health_flags_entropy_and_overdue_reseed():
    """Defect 7: health must notice stale, overdue, and low-entropy state."""
    state = instantiate(bytes([0x71]) * 32, b"health-nonce", b"health")
    assert check_health(state) == []

    for _ in range(9):
        state = generate(state, 8, b"health-load")["state"]

    issues = check_health(state)
    assert "RESEED_REQUIRED" in issues
    assert "STATE_STALE" in issues

    low_entropy = dict(state)
    low_entropy["entropy_count"] = 0
    assert "ENTROPY_LOW" in check_health(low_entropy)


def test_zero_length_generate_is_rejected_without_advancing():
    """Defect 8: zero-byte requests must raise and must not advance state."""
    state = instantiate(bytes([0x81]) * 32, b"zero-nonce", b"zero-length")
    snapshot = dict(state)

    with pytest.raises(ValueError):
        generate(state, 0, b"zero")

    assert state == snapshot
    assert len(generate(state, 1, b"after-zero")["output"]) == 1


def test_deserialize_rejects_generation_rollback():
    """Defect 9: an older valid blob must not roll the state backwards."""
    state = instantiate(bytes([0x91]) * 32, b"rollback-nonce", b"rollback")
    old_blob = serialize(state)
    newer = generate(state, 32, b"advance-for-rollback")["state"]
    new_blob = serialize(newer)

    restored = deserialize(new_blob)
    assert restored["generation"] == newer["generation"]

    with pytest.raises(ValueError):
        deserialize(old_blob)


def test_batch_requests_start_from_independent_state():
    """Defect 10: the second request must not continue the first one."""
    config = {
        "entropy": bytes([0xA1]) * 32,
        "nonce": b"batch-nonce",
        "personalization": b"batch-independent",
    }
    requests = [
        {"nbytes": 32, "additional": b"same-request"},
        {"nbytes": 32, "additional": b"same-request"},
    ]
    batch = drbg_batch(requests, config)
    standalone = generate(
        instantiate(
            config["entropy"],
            config["nonce"],
            config["personalization"],
        ),
        32,
        b"same-request",
    )["output"]

    assert len(batch["outputs"]) == 2
    assert batch["outputs"][0] == standalone
    assert batch["outputs"][1] == standalone
    assert len(batch["states"]) == 2