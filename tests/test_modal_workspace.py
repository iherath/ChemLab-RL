"""Workspace checks must follow an explicit target and reject account mismatches."""
import json
import os
import subprocess
import unittest
from unittest.mock import patch

from scripts import modal_train, upload_modal_data


class ModalWorkspaceTests(unittest.TestCase):
    def check_workspace(self, profiles, env, accepts):
        response = subprocess.CompletedProcess([], 0, json.dumps(profiles), "")
        for module in (modal_train, upload_modal_data):
            with self.subTest(module=module.__name__), patch.dict(os.environ, env, clear=True), \
                    patch("subprocess.run", return_value=response):
                if accepts:
                    module.verify_workspace()
                else:
                    with self.assertRaises((ValueError, RuntimeError)):
                        module.verify_workspace()

    def test_default_workspace_is_preserved(self):
        self.check_workspace([dict(name="partner", workspace="ish70", active=True)], {}, True)
        self.check_workspace([dict(name="mine", workspace="mliang4", active=True)], {}, False)

    def test_explicit_workspace_must_match(self):
        env = {"CHEMLAB_MODAL_WORKSPACE": "mliang4"}
        self.check_workspace([dict(name="mine", workspace="mliang4", active=True)], env, True)
        self.check_workspace([dict(name="partner", workspace="ish70", active=True)], env, False)
        self.check_workspace([], env, False)

    def test_selected_profile_is_checked_instead_of_active_profile(self):
        self.check_workspace([dict(name="partner", workspace="ish70", active=True),
                              dict(name="mine", workspace="mliang4", active=False)],
                             {"CHEMLAB_MODAL_WORKSPACE": "mliang4", "MODAL_PROFILE": "mine"}, True)

    def test_token_overrides_and_empty_target_are_rejected(self):
        profiles = [dict(name="mine", workspace="mliang4", active=True)]
        self.check_workspace(profiles, {"CHEMLAB_MODAL_WORKSPACE": ""}, False)
        self.check_workspace(profiles, {"CHEMLAB_MODAL_WORKSPACE": "mliang4",
                                        "MODAL_TOKEN_SECRET": "test-only"}, False)


if __name__ == "__main__":
    unittest.main()
