"""Deterministic random bit generator with an explicit, serialisable state machine.

The repository ships this module as eight stubs.  Every public function below
raises NotImplementedError until it is implemented.  The docstrings define the
byte-level contract; the tests in ``tests/`` are the executable part of that
contract.

State dictionary
----------------
A state is a plain dict with exactly these keys and value types::

    {
        "version": int,                    # 1
        "key": bytes,                      # 32 bytes
        "counter": int,                    # non-negative, 64-bit block counter
        "generation": int,                 # monotonically increasing revision
        "reseed_counter": int,             # generate calls since last reseed
        "entropy_count": int,              # total entropy bytes accumulated
        "last_reseed_generation": int,     # generation of last reseed / PR refresh
        "instance_id": bytes,              # 16-byte stable instance identifier
        "invalidated": bool,               # True once a PR refresh consumes it
    }

All concatenations in the derivation formulas below use the length prefix::

    LP(x) = len(x).to_bytes(4, "big") + x
    H(x)  = hashlib.sha256(x).digest()

``additional`` is always a bytes object (possibly empty), never None.

Instantiation
-------------
::

    key = H(b"sec-drbg/v1/instantiate" + LP(entropy) + LP(nonce)
            + LP(personalization))
    instance_id = H(b"sec-drbg/v1/instance" + LP(entropy) + LP(nonce)
                    + LP(personalization))[:16]

The initial counter, generation, reseed_counter and last_reseed_generation are
all zero.  At least 32 bytes of entropy are required; short input is an error,
never zero-padded.  ``nonce`` and ``personalization`` may be empty.

Generation
----------
For each 32-byte output block, with the block counter encoded big-endian::

    block = HMAC-SHA256(key, b"sec-drbg/v1/generate"
                        + counter.to_bytes(8, "big") + LP(additional))

The counter advances once per generated block, even when the final block is
truncated.  After all blocks, the returned state has::

    key = H(b"sec-drbg/v1/ratchet" + old_key + output)
    generation += 1
    reseed_counter += 1

where ``output`` is the exact returned (possibly truncated) byte string.  A
request for zero bytes, or for more than 4096 bytes, raises ValueError and
leaves the input state untouched.  Generate never mutates its input.

Reseeding
---------
Reseeding must mix the new entropy with the old state rather than replace it::

    key = H(b"sec-drbg/v1/reseed" + LP(old_key)
            + old_counter.to_bytes(8, "big") + LP(entropy) + LP(additional))

The returned state increments ``counter`` and ``generation`` by one, sets
``reseed_counter`` to zero, adds ``len(entropy)`` to ``entropy_count``, and
sets ``last_reseed_generation`` to the new generation.  The old state remains a
usable historical object; its future output differs from the reseeded state.

Prediction resistance
---------------------
``predict_resistance`` consumes 32 fresh bytes from ``os.urandom`` and derives::

    key = H(b"sec-drbg/v1/pr" + LP(old_key) + LP(fresh) + LP(additional))

It returns a new state with counter and generation advanced, reseed_counter
reset to zero, entropy_count increased by 32, and last_reseed_generation set to
the new generation.  It also marks the input state dict as invalidated.  Any
later use of that old dict raises ValueError.

Serialisation format
--------------------
The canonical JSON payload contains every state field, with bytes encoded as
base64 ASCII, and is produced with ``sort_keys=True`` and separators (",", ":").
The blob is::

    b"DRBG" + bytes([1]) + len(payload).to_bytes(4, "big")
    + payload + H(prefix + payload)

The trailing 32 bytes are the checksum over everything before the checksum.
Deserialisation rejects a bad magic, a version other than 1, a truncated or
inconsistent length, a checksum mismatch, malformed fields, and rollback.

Rollback high-water mark
------------------------
Deserialisation is a process-local state-machine entry point.  The module keeps
a high-water mark of the greatest generation restored for each instance_id.  A
blob whose generation is lower than that mark is a rollback and raises
ValueError.  Equal generations are accepted so that an ordinary serialise /
deserialise round trip is lossless.

Health checks
-------------
``check_health`` returns a list of stable issue codes, empty when healthy:

* STATE_INVALID       - the state was consumed by predict_resistance
* ENTROPY_LOW         - entropy_count is below 32
* RESEED_REQUIRED     - reseed_counter is greater than 8
* STATE_STALE         - generation minus last_reseed_generation is greater than 8

Batching
--------
``drbg_batch`` treats every request as an independent instance.  A request is a
dict with nbytes (required), additional (optional), and optional entropy, nonce
or personalization overrides.  The config dict supplies the defaults.  No state
is shared between requests.  The result is::

    {"outputs": [bytes, ...], "states": [state, ...]}
"""

import base64
import hashlib
import hmac
import json
import os

VERSION = 1
MIN_ENTROPY = 32
MAX_GENERATE_WITHOUT_RESEED = 8
MAX_REQUEST_BYTES = 4096
HEADER = b"DRBG"
TAG_SIZE = 32

_STATE_FIELDS = frozenset(
    {
        "version",
        "key",
        "counter",
        "generation",
        "reseed_counter",
        "entropy_count",
        "last_reseed_generation",
        "instance_id",
        "invalidated",
    }
)
_INT_FIELDS = (
    "version",
    "counter",
    "generation",
    "reseed_counter",
    "entropy_count",
    "last_reseed_generation",
)
_UINT64_MAX = (1 << 64) - 1

# Process-local rollback high-water mark: instance_id -> greatest generation
# ever restored via deserialize in this process.
_ROLLBACK_MARK: dict[bytes, int] = {}


def _lp(data: bytes) -> bytes:
    """Length-prefix a byte string: len(data) as 4 big-endian bytes + data."""
    return len(data).to_bytes(4, "big") + data


def _digest(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _require_bytes(value, name: str) -> None:
    if not isinstance(value, bytes):
        raise ValueError(f"{name} must be bytes, got {type(value).__name__}")


def _validate_state(state: dict, allow_invalidated: bool = False) -> None:
    """Validate the shape and value types of a state dict.

    Raises ValueError for any malformed state and for consumed states unless
    ``allow_invalidated`` is set (used by check_health, which reports the
    condition instead of refusing to run).
    """
    if not isinstance(state, dict):
        raise ValueError("state must be a dict")
    if set(state.keys()) != _STATE_FIELDS:
        raise ValueError("state has missing or unknown fields")
    if state["version"] != VERSION:
        raise ValueError(f"unsupported state version: {state['version']!r}")
    if not isinstance(state["key"], bytes) or len(state["key"]) != 32:
        raise ValueError("state key must be exactly 32 bytes")
    if not isinstance(state["instance_id"], bytes) or len(state["instance_id"]) != 16:
        raise ValueError("instance_id must be exactly 16 bytes")
    if not isinstance(state["invalidated"], bool):
        raise ValueError("invalidated must be a bool")
    for field in _INT_FIELDS:
        value = state[field]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field} must be an int")
        if value < 0:
            raise ValueError(f"{field} must be non-negative")
    if state["counter"] > _UINT64_MAX:
        raise ValueError("counter exceeds the 64-bit block-counter range")
    if state["last_reseed_generation"] > state["generation"]:
        raise ValueError("last_reseed_generation cannot exceed generation")
    if not allow_invalidated and state["invalidated"]:
        raise ValueError("state has been invalidated by a prediction-resistance refresh")


def instantiate(entropy: bytes, nonce: bytes, personalization: bytes) -> dict:
    """Create a fresh state from at least 32 bytes of entropy.

    Raises ValueError when entropy is shorter than MIN_ENTROPY or when an
    argument is not bytes.  Returns the state dict described in the module
    docstring.
    """
    # Defect 6 fix: insufficient entropy is an error, never zero-padded.
    _require_bytes(entropy, "entropy")
    _require_bytes(nonce, "nonce")
    _require_bytes(personalization, "personalization")
    if len(entropy) < MIN_ENTROPY:
        raise ValueError(
            f"at least {MIN_ENTROPY} bytes of entropy are required, "
            f"got {len(entropy)}"
        )

    key = _digest(
        b"sec-drbg/v1/instantiate"
        + _lp(entropy)
        + _lp(nonce)
        + _lp(personalization)
    )
    instance_id = _digest(
        b"sec-drbg/v1/instance"
        + _lp(entropy)
        + _lp(nonce)
        + _lp(personalization)
    )[:16]

    return {
        "version": VERSION,
        "key": key,
        "counter": 0,
        "generation": 0,
        "reseed_counter": 0,
        "entropy_count": len(entropy),
        "last_reseed_generation": 0,
        "instance_id": instance_id,
        "invalidated": False,
    }


def generate(state: dict, nbytes: int, additional: bytes) -> dict:
    """Generate nbytes from state and return {"state": new_state, "output": bytes}.

    The request is rejected with ValueError for nbytes outside 1..4096 or for
    an invalidated state.  The returned state is a fresh dict; the input is not
    mutated.
    """
    _validate_state(state)
    _require_bytes(additional, "additional")
    if isinstance(nbytes, bool) or not isinstance(nbytes, int):
        raise ValueError("nbytes must be an int")
    # Defect 8 fix: zero-length (and overlong) requests are rejected before any
    # state movement, so the call sequence cannot be desynchronised.
    if nbytes <= 0 or nbytes > MAX_REQUEST_BYTES:
        raise ValueError(
            f"nbytes must be within 1..{MAX_REQUEST_BYTES}, got {nbytes}"
        )

    key = state["key"]
    counter = state["counter"]
    blocks = (nbytes + 31) // 32
    if counter + blocks > _UINT64_MAX:
        raise ValueError("block counter would overflow its 64-bit range")

    # Defect 2 fix: every block consumes and advances the block counter, so
    # repeated calls produce distinct output instead of replaying bytes.
    output = bytearray()
    for _ in range(blocks):
        message = (
            b"sec-drbg/v1/generate"
            + counter.to_bytes(8, "big")
            + _lp(additional)
        )
        output.extend(hmac.new(key, message, hashlib.sha256).digest())
        counter += 1
    output = bytes(output[:nbytes])

    # Defect 2 fix (cont.): after output, the key is ratcheted and the revision
    # counters advance in a fresh state; the input state is left untouched.
    new_state = dict(state)
    new_state["key"] = _digest(b"sec-drbg/v1/ratchet" + key + output)
    new_state["counter"] = counter
    new_state["generation"] = state["generation"] + 1
    new_state["reseed_counter"] = state["reseed_counter"] + 1
    return {"state": new_state, "output": output}


def reseed(state: dict, entropy: bytes, additional: bytes) -> dict:
    """Mix new entropy into the old state and return the advanced state.

    Short entropy, an invalidated state, or a non-bytes argument raises
    ValueError.  Entropy is accumulated, never used as a replacement for the
    old key material.  The old state dict is not invalidated.
    """
    _validate_state(state)
    _require_bytes(entropy, "entropy")
    _require_bytes(additional, "additional")
    # Defect 6 applies here too: short entropy is rejected, never padded.
    if len(entropy) < MIN_ENTROPY:
        raise ValueError(
            f"at least {MIN_ENTROPY} bytes of entropy are required, "
            f"got {len(entropy)}"
        )

    # Defect 1 fix: derive the new key from BOTH the old state material and the
    # fresh entropy instead of replacing the state outright.
    new_key = _digest(
        b"sec-drbg/v1/reseed"
        + _lp(state["key"])
        + state["counter"].to_bytes(8, "big")
        + _lp(entropy)
        + _lp(additional)
    )

    new_generation = state["generation"] + 1
    new_state = dict(state)
    new_state["key"] = new_key
    new_state["counter"] = state["counter"] + 1
    new_state["generation"] = new_generation
    new_state["reseed_counter"] = 0
    new_state["entropy_count"] = state["entropy_count"] + len(entropy)
    new_state["last_reseed_generation"] = new_generation
    # The old dict stays valid and usable as a historical object.
    return new_state


def serialize(state: dict) -> bytes:
    """Serialise state to the versioned, checksummed blob format.

    An invalidated state or a state with unknown fields/version raises
    ValueError.
    """
    _validate_state(state)

    payload_dict = {field: state[field] for field in sorted(_STATE_FIELDS)}
    payload = json.dumps(
        payload_dict,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda value: base64.b64encode(value).decode("ascii")
        if isinstance(value, bytes)
        else _raise_serialize_type(value),
    ).encode("utf-8")

    # Defect 3 fix: magic and an explicit format version are written, followed
    # by a length field, so cross-version parsing cannot silently misalign.
    prefix = HEADER + bytes([VERSION]) + len(payload).to_bytes(4, "big")
    # Defect 4 fix: a SHA-256 checksum covers every byte preceding the tag.
    checksum = _digest(prefix + payload)
    return prefix + payload + checksum


def _raise_serialize_type(value):
    raise ValueError(f"cannot serialise value of type {type(value).__name__}")


def deserialize(blob: bytes) -> dict:
    """Parse a serialised state, rejecting corruption and rollback.

    ValueError is raised for a bad magic, unsupported version, length
    mismatch, checksum failure, malformed payload, invalidated state, or a
    generation below the process-local high-water mark for that instance.
    """
    if not isinstance(blob, bytes):
        raise ValueError("blob must be bytes")
    # header(4) + version(1) + length(4) + checksum(32), payload may be empty
    # only in theory; a real payload is longer.
    if len(blob) < 4 + 1 + 4 + TAG_SIZE:
        raise ValueError("blob is truncated")
    if blob[:4] != HEADER:
        raise ValueError("bad magic header")
    version = blob[4]
    # Defect 3 fix (read side): only the supported version is accepted.
    if version != VERSION:
        raise ValueError(f"unsupported blob version: {version}")
    payload_length = int.from_bytes(blob[5:9], "big")
    if len(blob) != 9 + payload_length + TAG_SIZE:
        raise ValueError("declared payload length does not match blob size")

    payload = blob[9 : 9 + payload_length]
    stored_checksum = blob[9 + payload_length :]
    # Defect 4 fix: verify the checksum before trusting any payload content.
    expected_checksum = _digest(blob[: 9 + payload_length])
    if not hmac.compare_digest(stored_checksum, expected_checksum):
        raise ValueError("checksum mismatch: blob is corrupted or tampered")

    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("malformed payload: not valid JSON") from exc
    if not isinstance(raw, dict) or set(raw.keys()) != _STATE_FIELDS:
        raise ValueError("malformed payload: missing or unknown fields")

    try:
        state = {
            "version": int(raw["version"]),
            "key": _b64_decode_field(raw["key"], "key"),
            "counter": int(raw["counter"]),
            "generation": int(raw["generation"]),
            "reseed_counter": int(raw["reseed_counter"]),
            "entropy_count": int(raw["entropy_count"]),
            "last_reseed_generation": int(raw["last_reseed_generation"]),
            "instance_id": _b64_decode_field(raw["instance_id"], "instance_id"),
            "invalidated": bool(raw["invalidated"]),
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(f"malformed payload field: {exc}") from exc

    # Full structural validation, including rejection of consumed states.
    _validate_state(state)

    # Defect 9 fix: a per-instance high-water mark forbids restoring a state
    # older than one already restored in this process; equal generation is
    # allowed so a plain round trip stays lossless.
    instance_id = state["instance_id"]
    generation = state["generation"]
    high_water = _ROLLBACK_MARK.get(instance_id)
    if high_water is not None and generation < high_water:
        raise ValueError(
            "rollback rejected: generation "
            f"{generation} is below restored high-water mark {high_water}"
        )
    _ROLLBACK_MARK[instance_id] = generation
    return state


def _b64_decode_field(value, name: str) -> bytes:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a base64 string")
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise ValueError(f"{name} is not valid base64") from exc


def predict_resistance(state: dict, additional: bytes) -> dict:
    """Refresh with fresh OS entropy and invalidate the supplied old state.

    Returns a new state.  The input dict is marked invalidated in place; any
    subsequent generate, reseed or predict_resistance call with it raises
    ValueError.  An already invalidated or malformed input raises ValueError.
    """
    _validate_state(state)
    _require_bytes(additional, "additional")

    fresh = os.urandom(MIN_ENTROPY)
    new_key = _digest(
        b"sec-drbg/v1/pr"
        + _lp(state["key"])
        + _lp(fresh)
        + _lp(additional)
    )

    new_generation = state["generation"] + 1
    new_state = dict(state)
    new_state["key"] = new_key
    new_state["counter"] = state["counter"] + 1
    new_state["generation"] = new_generation
    new_state["reseed_counter"] = 0
    new_state["entropy_count"] = state["entropy_count"] + MIN_ENTROPY
    new_state["last_reseed_generation"] = new_generation
    new_state["invalidated"] = False

    # Defect 5 fix: the old state is consumed in place and can never advance
    # again.  Only the returned state carries a usable continuation.
    state["invalidated"] = True
    return new_state


def check_health(state: dict) -> list[str]:
    """Return stable health issue codes for state, in the documented order."""
    # Allow invalidated states through so STATE_INVALID can be reported.
    _validate_state(state, allow_invalidated=True)

    issues: list[str] = []
    if state["invalidated"]:
        issues.append("STATE_INVALID")
    if state["entropy_count"] < MIN_ENTROPY:
        issues.append("ENTROPY_LOW")
    # Defect 7 fix: both the generate count since the last reseed and the
    # generation gap since the last reseed/PR refresh are monitored.
    if state["reseed_counter"] > MAX_GENERATE_WITHOUT_RESEED:
        issues.append("RESEED_REQUIRED")
    if (
        state["generation"] - state["last_reseed_generation"]
        > MAX_GENERATE_WITHOUT_RESEED
    ):
        issues.append("STATE_STALE")
    return issues


def drbg_batch(requests: list[dict], config: dict) -> dict:
    """Run independent requests and return outputs plus final states.

    Each request starts from a fresh instantiate call built from config
    defaults plus request-level overrides.  No DRBG state is shared between
    requests.  Malformed input raises ValueError.
    """
    if not isinstance(requests, list):
        raise ValueError("requests must be a list")
    if not isinstance(config, dict):
        raise ValueError("config must be a dict")

    defaults = {
        "entropy": config.get("entropy"),
        "nonce": config.get("nonce", b""),
        "personalization": config.get("personalization", b""),
    }
    if defaults["entropy"] is None:
        raise ValueError("config must provide default entropy")
    for name in ("entropy", "nonce", "personalization"):
        _require_bytes(defaults[name], f"config.{name}")

    outputs: list[bytes] = []
    states: list[dict] = []
    for index, request in enumerate(requests):
        if not isinstance(request, dict):
            raise ValueError(f"request {index} must be a dict")
        if "nbytes" not in request:
            raise ValueError(f"request {index} is missing nbytes")

        entropy = request.get("entropy", defaults["entropy"])
        nonce = request.get("nonce", defaults["nonce"])
        personalization = request.get(
            "personalization", defaults["personalization"]
        )
        additional = request.get("additional", b"")
        for name, value in (
            ("entropy", entropy),
            ("nonce", nonce),
            ("personalization", personalization),
            ("additional", additional),
        ):
            _require_bytes(value, f"request {index}.{name}")

        # Defect 10 fix: every request gets its own freshly instantiated
        # state, so no output stream can continue a previous request's state.
        local_state = instantiate(entropy, nonce, personalization)
        result = generate(local_state, request["nbytes"], additional)
        outputs.append(result["output"])
        states.append(result["state"])

    return {"outputs": outputs, "states": states}
