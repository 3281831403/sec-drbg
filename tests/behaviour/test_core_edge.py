"""Boundary and recovery tests for the DRBG state machine."""

from drbg.core import (
    deserialize,
    drbg_batch,
    generate,
    instantiate,
    reseed,
    serialize,
)

FIXED_ENTROPY = bytes(range(32))
FIXED_NONCE = b"fixed-nonce-0001"
FIXED_PERSONALIZATION = b"sec-drbg-vector"
FIXED_ADDITIONAL = b"vector-additional"
FIXED_FIRST_32 = bytes.fromhex(
    "d4c0e2d576e9ff4b06dd8dbe6a1d1489"
    "6a67618fa9f60d66d47c37f0ba84d50d"
)


def test_fixed_vector_first_32_bytes():
    """The first output under the documented derivation is an exact vector."""
    state = instantiate(FIXED_ENTROPY, FIXED_NONCE, FIXED_PERSONALIZATION)
    output = generate(state, 32, FIXED_ADDITIONAL)["output"]
    assert output == FIXED_FIRST_32


def test_consecutive_generation_is_not_repeated():
    """Two back-to-back requests at a fixed boundary must diverge."""
    state = instantiate(bytes([0xB1]) * 32, b"edge-nonce", b"edge-sequence")
    first = generate(state, 32, b"edge-additional")
    second = generate(first["state"], 32, b"edge-additional")

    assert len(first["output"]) == 32
    assert len(second["output"]) == 32
    assert first["output"] != second["output"]


def test_serialization_roundtrip_preserves_continuation():
    """Deserialised state must continue exactly like the uninterrupted state."""
    state = instantiate(
        bytes([0xC1]) * 32,
        b"roundtrip-nonce",
        b"roundtrip-preserve",
    )
    blob = serialize(state)
    restored = deserialize(blob)

    assert restored == state
    uninterrupted = generate(state, 48, b"roundtrip-additional")["output"]
    resumed = generate(restored, 48, b"roundtrip-additional")["output"]
    assert resumed == uninterrupted


def test_reseeded_state_differs_from_old_state():
    """Old state can still advance but must not replay the reseeded stream."""
    state = instantiate(
        bytes([0xD1]) * 32,
        b"old-state-nonce",
        b"old-state-after-reseed",
    )
    reseeded = reseed(state, bytes([0xD2]) * 32, b"reseed-separator")

    old_bytes = generate(state, 32, b"after-reseed")["output"]
    new_bytes = generate(reseeded, 32, b"after-reseed")["output"]

    assert reseeded["entropy_count"] == 64
    assert old_bytes != new_bytes


def test_batch_matches_individual_generation():
    """Independent batches and per-request standalone generation must agree."""
    config = {
        "entropy": bytes([0xE1]) * 32,
        "nonce": b"edge-batch-nonce",
        "personalization": b"edge-batch-independent",
    }
    requests = [
        {"nbytes": 16, "additional": b"request-one"},
        {"nbytes": 32, "additional": b"request-two"},
    ]

    first_batch = drbg_batch(requests, config)
    second_batch = drbg_batch(requests, config)
    assert first_batch["outputs"] == second_batch["outputs"]

    for request, batch_output in zip(requests, first_batch["outputs"]):
        standalone = generate(
            instantiate(
                config["entropy"],
                config["nonce"],
                config["personalization"],
            ),
            request["nbytes"],
            request["additional"],
        )["output"]
        assert batch_output == standalone