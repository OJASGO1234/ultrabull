"""Offline tests for the Groww auth path used by api.live (no network, no creds).

growwapi 1.5.0: GrowwAPI.get_access_token(api_key, totp=None, secret=None).
For the TOTP flow, `api_key` is the *TOTP token* (JWT from Groww's API-keys
page) and `totp` is the 6-digit code from the base32 *TOTP secret*.
"""
import base64
import json
import time

import pyotp
import pytest
from fastapi import HTTPException

from api import live

ENV = ("groww_access_token", "groww_api_key", "groww_totp_gen", "groww_totp_token", "groww_totp_secret")


def _jwt(exp_offset):
    b = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).decode().rstrip("=")
    return f"{b({'alg': 'none'})}.{b({'exp': time.time() + exp_offset})}.sig"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for v in ENV:
        monkeypatch.delenv(v, raising=False)


@pytest.fixture
def fake_api(monkeypatch):
    import growwapi
    calls = []

    class FakeAPI:
        def __init__(self, token):
            self.token = token

        @staticmethod
        def get_access_token(*a, **k):
            calls.append((a, k))
            return "minted"

    monkeypatch.setattr(growwapi, "GrowwAPI", FakeAPI)
    return calls


def test_sdk_requires_totp_or_secret():
    from growwapi import GrowwAPI
    with pytest.raises(ValueError):
        GrowwAPI.get_access_token("k")  # key alone is rejected before any request


def test_totp_token_and_secret_mint_token(monkeypatch, fake_api):
    seed = pyotp.random_base32()
    tok = _jwt(10_000)
    monkeypatch.setenv("groww_totp_token", tok)
    monkeypatch.setenv("groww_totp_secret", seed)
    assert live._new_client().token == "minted"
    (args, kwargs), = fake_api
    assert args[0] == tok                      # TOTP token goes in the api_key slot
    assert len(args[1]) == 6 and args[1].isdigit()  # 6-digit code, never the seed
    assert kwargs == {}


def test_totp_wins_over_fresh_access_token(monkeypatch, fake_api):
    monkeypatch.setenv("groww_access_token", _jwt(10_000))
    monkeypatch.setenv("groww_totp_token", _jwt(10_000))
    monkeypatch.setenv("groww_totp_secret", pyotp.random_base32())
    assert live._new_client().token == "minted"
    assert len(fake_api) == 1


def test_expired_access_token_does_not_block_totp(monkeypatch, fake_api):
    monkeypatch.setenv("groww_access_token", _jwt(-10))
    monkeypatch.setenv("groww_totp_token", _jwt(10_000))
    monkeypatch.setenv("groww_totp_secret", pyotp.random_base32())
    assert live._new_client().token == "minted"


def test_legacy_layout_api_key_plus_base32_totp_gen(monkeypatch, fake_api):
    monkeypatch.setenv("groww_api_key", "legacy-key")
    monkeypatch.setenv("groww_totp_gen", pyotp.random_base32())
    assert live._new_client().token == "minted"
    assert fake_api[0][0][0] == "legacy-key"


def test_jwt_in_totp_gen_is_token_not_secret(monkeypatch, fake_api):
    monkeypatch.setenv("groww_api_key", _jwt(10_000))
    monkeypatch.setenv("groww_totp_gen", _jwt(10_000))  # not a seed: must not reach pyotp
    with pytest.raises(HTTPException) as e:
        live._new_client()
    assert e.value.status_code == 503 and fake_api == []


def test_jwt_totp_gen_plus_secret_uses_it_as_token(monkeypatch, fake_api):
    tok = _jwt(10_000)
    monkeypatch.setenv("groww_totp_gen", tok)
    monkeypatch.setenv("groww_totp_secret", pyotp.random_base32())
    live._new_client()
    assert fake_api[0][0][0] == tok


def test_fresh_access_token_alone_is_fallback(monkeypatch, fake_api):
    tok = _jwt(10_000)
    monkeypatch.setenv("groww_access_token", tok)
    assert live._new_client().token == tok
    assert fake_api == []


def test_opaque_access_token_alone_is_used(monkeypatch, fake_api):
    monkeypatch.setenv("groww_access_token", "pre-token")  # non-JWT: cannot judge expiry
    assert live._new_client().token == "pre-token"


def test_expired_access_token_alone_raises_503(monkeypatch, fake_api):
    monkeypatch.setenv("groww_access_token", _jwt(-10))
    with pytest.raises(HTTPException) as e:
        live._new_client()
    assert e.value.status_code == 503 and "expired" in e.value.detail
    assert fake_api == []


def test_blank_access_token_ignored(monkeypatch, fake_api):
    monkeypatch.setenv("groww_access_token", "   ")
    with pytest.raises(HTTPException) as e:
        live._new_client()
    assert e.value.status_code == 503


def test_no_credentials_raises_503():
    with pytest.raises(HTTPException) as e:
        live._new_client()
    assert e.value.status_code == 503


def test_token_only_without_secret_raises_503(monkeypatch, fake_api):
    monkeypatch.setenv("groww_totp_token", _jwt(10_000))
    with pytest.raises(HTTPException) as e:
        live._new_client()
    assert e.value.status_code == 503 and fake_api == []


def test_mint_failure_is_safe(monkeypatch, caplog):
    import growwapi
    seed, tok = pyotp.random_base32(), _jwt(10_000)
    monkeypatch.setenv("groww_totp_token", tok)
    monkeypatch.setenv("groww_totp_secret", seed)

    class FakeAPI:
        @staticmethod
        def get_access_token(*a, **k):
            raise RuntimeError(f"boom Bearer {a[0]} {a[1]}")

    monkeypatch.setattr(growwapi, "GrowwAPI", FakeAPI)
    with caplog.at_level("DEBUG"), pytest.raises(HTTPException) as e:
        live._new_client()
    assert e.value.status_code == 503 and "RuntimeError" in e.value.detail
    blob = e.value.detail + caplog.text
    assert tok not in blob and seed not in blob and "Bearer" not in blob


def test_bad_secret_is_handled_without_leak(monkeypatch, fake_api):
    monkeypatch.setenv("groww_totp_token", _jwt(10_000))
    monkeypatch.setenv("groww_totp_secret", "not base32 !!")
    with pytest.raises(HTTPException) as e:
        live._new_client()
    assert e.value.status_code == 503 and "not base32" not in e.value.detail
