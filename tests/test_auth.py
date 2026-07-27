import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attrition_predictor.auth import (
    DEFAULT_PASSWORD_HASH,
    DEFAULT_USERNAME,
    hash_password,
    resolve_credentials,
    verify_credentials,
)


class TestHashPassword(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(hash_password("hello"), hash_password("hello"))

    def test_different_inputs_different_hashes(self):
        self.assertNotEqual(hash_password("hello"), hash_password("world"))

    def test_returns_hex_string(self):
        h = hash_password("hello")
        self.assertEqual(len(h), 64)
        int(h, 16)  # raises ValueError if not valid hex


class TestResolveCredentials(unittest.TestCase):
    def test_falls_back_to_default_when_no_secrets(self):
        self.assertEqual(resolve_credentials(None), (DEFAULT_USERNAME, DEFAULT_PASSWORD_HASH))
        self.assertEqual(resolve_credentials({}), (DEFAULT_USERNAME, DEFAULT_PASSWORD_HASH))

    def test_uses_configured_secrets_when_present(self):
        custom_hash = hash_password("something-else")
        result = resolve_credentials({"username": "hr_admin", "password_hash": custom_hash})
        self.assertEqual(result, ("hr_admin", custom_hash))

    def test_falls_back_when_secrets_incomplete(self):
        # missing password_hash -> falls back entirely, doesn't half-apply
        result = resolve_credentials({"username": "hr_admin"})
        self.assertEqual(result, (DEFAULT_USERNAME, DEFAULT_PASSWORD_HASH))


class TestVerifyCredentials(unittest.TestCase):
    def test_correct_default_credentials_pass(self):
        self.assertTrue(
            verify_credentials("admin", "attrition2026", DEFAULT_USERNAME, DEFAULT_PASSWORD_HASH)
        )

    def test_wrong_password_fails(self):
        self.assertFalse(
            verify_credentials("admin", "wrong-password", DEFAULT_USERNAME, DEFAULT_PASSWORD_HASH)
        )

    def test_wrong_username_fails(self):
        self.assertFalse(
            verify_credentials("someone-else", "attrition2026", DEFAULT_USERNAME, DEFAULT_PASSWORD_HASH)
        )

    def test_empty_username_fails(self):
        self.assertFalse(
            verify_credentials("", "attrition2026", DEFAULT_USERNAME, DEFAULT_PASSWORD_HASH)
        )

    def test_empty_password_fails(self):
        self.assertFalse(
            verify_credentials("admin", "", DEFAULT_USERNAME, DEFAULT_PASSWORD_HASH)
        )


if __name__ == "__main__":
    unittest.main()
