import unittest
import sys
from pathlib import Path

# Add repo root to path to allow importing dsstar
sys.path.append(str(Path(__file__).parent.parent))

from dsstar import DSConfig, create_config
from dataclasses import dataclass

@dataclass
class MockArgs:
    resume: str = None
    interactive: bool = False
    max_rounds: int = None
    config: str = "config.yaml"
    edit_last: bool = False
    data_files: list = None
    query: str = None

class TestConfig(unittest.TestCase):
    def test_create_config_merging(self):
        file_config = {
            'model_name': 'test-model',
            'api_key': 'secret-key',
            'max_refinement_rounds': 10
        }
        args = MockArgs(interactive=True, max_rounds=20)

        config = create_config(args, file_config)

        self.assertEqual(config.model_name, 'test-model')
        self.assertEqual(config.api_key, 'secret-key')
        self.assertEqual(config.max_refinement_rounds, 20) # Args override file
        self.assertTrue(config.interactive) # Args override default

if __name__ == '__main__':
    unittest.main()
