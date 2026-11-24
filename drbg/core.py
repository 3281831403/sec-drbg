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

VERSION = 1
MIN_ENTROPY = 32
MAX_GENERATE_WITHOUT_RESEED = 8
MAX_REQUEST_BYTES = 4096
HEADER = b"DRBG"
TAG_SIZE = 32


def instantiate(entropy: bytes, nonce: bytes, personalization: bytes) -> dict:
    """Create a fresh state from at least 32 bytes of entropy.

    Raises ValueError when entropy is shorter than MIN_ENTROPY or when an
    argument is not bytes.  Returns the state dict described in the module
    docstring.
    """
    raise NotImplementedError("instantiate is not implemented")


def generate(state: dict, nbytes: int, additional: bytes) -> dict:
    """Generate nbytes from state and return {"state": new_state, "output": bytes}.

    The request is rejected with ValueError for nbytes outside 1..4096 or for
    an invalidated state.  The returned state is a fresh dict; the input is not
    mutated.
    """
    raise NotImplementedError("generate is not implemented")


def reseed(state: dict, entropy: bytes, additional: bytes) -> dict:
    """Mix new entropy into the old state and return the advanced state.

    Short entropy, an invalidated state, or a non-bytes argument raises
    ValueError.  Entropy is accumulated, never used as a replacement for the
    old key material.  The old state dict is not invalidated.
    """
    raise NotImplementedError("reseed is not implemented")


def serialize(state: dict) -> bytes:
    """Serialise state to the versioned, checksummed blob format.

    An invalidated state or a state with unknown fields/version raises
    ValueError.
    """
    raise NotImplementedError("serialize is not implemented")


def deserialize(blob: bytes) -> dict:
    """Parse a serialised state, rejecting corruption and rollback.

    ValueError is raised for a bad magic, unsupported version, length
    mismatch, checksum failure, malformed payload, invalidated state, or a
    generation below the process-local high-water mark for that instance.
    """
    raise NotImplementedError("deserialize is not implemented")


def predict_resistance(state: dict, additional: bytes) -> dict:
    """Refresh with fresh OS entropy and invalidate the supplied old state.

    Returns a new state.  The input dict is marked invalidated in place; any
    subsequent generate, reseed or predict_resistance call with it raises
    ValueError.  An already invalidated or malformed input raises ValueError.
    """
    raise NotImplementedError("predict_resistance is not implemented")


def check_health(state: dict) -> list[str]:
    """Return stable health issue codes for state, in the documented order."""
    raise NotImplementedError("check_health is not implemented")


def drbg_batch(requests: list[dict], config: dict) -> dict:
    """Run independent requests and return outputs plus final states.

    Each request starts from a fresh instantiate call built from config
    defaults plus request-level overrides.  No DRBG state is shared between
    requests.  Malformed input raises ValueError.
    """
    raise NotImplementedError("drbg_batch is not implemented")