import inspect
import unittest

from axanctum.scanner import scan_coin


class ScanCoinSourceSafetyTest(unittest.TestCase):
    def test_scan_coin_does_not_read_batch_candidate_variable_in_return_payload(self):
        source = inspect.getsource(scan_coin)

        self.assertNotIn('r.get("short_regime_reasons"', source)
        self.assertNotIn("r.get('short_regime_reasons'", source)


if __name__ == "__main__":
    unittest.main()
