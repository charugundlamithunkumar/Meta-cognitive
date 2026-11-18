import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from audit_allocations import scan


class TestAllocationAudit(unittest.TestCase):
    def test_archive_metadata_and_conservative_exposure(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base/'real').mkdir()
            (base/'real/config.json').write_text(json.dumps(dict(ids=[52, 11000])))
            with tarfile.open(base/'archived.tar.gz', 'w:gz') as archive:
                for name, raw in [('arm/config.json', b'{"ids":[53],"mock":false}'),
                                  ('arm/rows.jsonl', b'NOT JSON; MUST NOT BE PARSED')]:
                    member = tarfile.TarInfo(name)
                    member.size = len(raw)
                    archive.addfile(member, io.BytesIO(raw))
            got = scan([base], base)
            self.assertEqual(got['reserve_ids_found'], [52, 53])
            self.assertEqual(got['historically_reserved_fresh_seeds_found'], [1000])
            self.assertEqual(got['errors'], [])

    def test_mock_does_not_count_as_model_exposure(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base/'config.json').write_text('{"ids":[52],"mock":true}')
            got = scan([base], base)
            self.assertEqual(got['reserve_ids_found'], [])
            self.assertEqual(len(got['records']), 1)

    def test_invalid_metadata_disclosed(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base/'config.json').write_text('{"ids":"52-99"}')
            got = scan([base], base)
            self.assertEqual(len(got['errors']), 1)
            self.assertEqual(got['records'], [])


if __name__ == '__main__':
    unittest.main()
