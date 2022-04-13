import unittest
import os

if __name__ == "__main__":
    loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    test_suite.addTests(
        loader.loadTestsFromModule(importlib.import_module("test_expanded_test_case"))
    )
    for entry in os.scandir():
        if entry.is_dir() and os.path.isfile(entry.name + "/__init__.py"):
            test_suite.addTests(loader.discover(entry.name, top_level_dir="."))
    runner = unittest.TextTestRunner()
    runner.run(test_suite)
