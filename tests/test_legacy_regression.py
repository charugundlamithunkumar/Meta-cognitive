"""Exercise archived regression tests for the shared primitives still in use."""
import importlib.util
import sys
from pathlib import Path


def load_tests(loader, tests, pattern):
    root = Path(__file__).resolve().parent.parent / 'previous work/tests'
    for name in ['test_core', 'test_arms_mock']:
        spec = importlib.util.spec_from_file_location('archived_' + name, root / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        original_path = list(sys.path)
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path[:] = original_path
        tests.addTests(loader.loadTestsFromModule(module))
    return tests
