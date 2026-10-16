"""Read-only cursor harness regression tests, using isolated transcripts."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/check_capture_cursors.py'
spec = importlib.util.spec_from_file_location('cursor_harness', SCRIPT)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


class CursorHarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.transcript = self.root / 'transcript.jsonl'
        self.transcript.write_bytes(b'{"a":1}\n{"b":2}\n')
        self.job = dict(provider='codex', session_id='session', transcript=str(self.transcript))
        self.cursor = self.root / f'.membrick/codex-state/capture-cursors/{h.cursor_key("session")}.json'
        self.cursor.parent.mkdir(parents=True)
        self.set_offset(0)

    def set_offset(self, value):
        self.cursor.write_text(json.dumps(dict(offset=value)))

    def check(self, validate=False):
        return h.check(self.root, self.job, validate)

    def cli(self, *extra):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = h.main(['--project', str(self.root), '--provider', 'codex',
                           '--session-id', 'session', '--transcript', str(self.transcript),
                           '--json', *extra])
        return code, [json.loads(line) for line in output.getvalue().splitlines()]

    def test_known_rust_hash_and_normalization(self):
        self.assertEqual(h.cursor_key('01a13788-b33a-7831-892f-1c194ac001c3'), '46b75b214da4a0f3')
        self.assertEqual(h.cursor_key(' A  B '), h.cursor_key('a b'))

    def test_zero_boundary_and_eof(self):
        for offset in (0, 8, 16):
            with self.subTest(offset=offset):
                self.set_offset(offset)
                result = self.check()
                self.assertTrue(result['ok'])
                self.assertEqual(result['caught_up'], offset == 16)
                self.assertEqual(result['uncaptured_complete_bytes'], 16 - offset)

    def test_bad_offsets(self):
        for offset in (1, 7, 17, -1, True, '8', None, 1.5):
            with self.subTest(offset=offset):
                self.set_offset(offset)
                self.assertFalse(self.check()['ok'])

    def test_mid_utf8_is_invalid(self):
        self.transcript.write_bytes('"中文"\n'.encode())
        self.set_offset(3)
        self.assertFalse(self.check()['ok'])

    def test_missing_and_corrupt_cursor_never_defaults_to_zero(self):
        self.cursor.unlink()
        self.assertFalse(self.check()['ok'])
        for content in ('{', '[]', '{}'):
            self.cursor.write_text(content)
            self.assertFalse(self.check()['ok'])

    def test_partial_tail_not_captured_or_validated(self):
        self.transcript.write_bytes(b'{}\n{"unfinished')
        self.set_offset(3)
        result = self.check(True)
        self.assertTrue(result['ok'])
        self.assertEqual(result['state'], 'waiting-for-newline')
        self.assertEqual(result['json_records_validated'], 1)
        self.set_offset(self.transcript.stat().st_size)
        self.assertFalse(self.check()['ok'])

    def test_large_partial_tail_backward_scan(self):
        self.transcript.write_bytes(b'{}\n' + b'x' * (h.BLOCK_BYTES * 2))
        self.assertEqual(self.check()['complete_bytes'], 3)

    def test_crlf_and_empty_transcript(self):
        self.transcript.write_bytes(b'{}\r\n')
        self.set_offset(4)
        self.assertTrue(self.check(True)['caught_up'])
        self.transcript.write_bytes(b'')
        self.set_offset(0)
        self.assertTrue(self.check(True)['caught_up'])

    def test_bad_complete_json_record_reports_byte_offset(self):
        self.transcript.write_bytes(b'{}\ninvalid\n')
        result = self.check(True)
        self.assertFalse(result['ok'])
        self.assertIn('byte 3', result['errors'][0])

    def test_discovery_and_bad_job(self):
        directory = self.root / '.membrick/runtime/transcript-captures'
        directory.mkdir(parents=True)
        for provider in h.PROVIDERS:
            record = dict(request=dict(provider=provider, input=dict(session_id='session',
                          transcript_path=str(self.transcript))), target_bytes=16)
            (directory / f'{provider}-{h.cursor_key("session")}.json').write_text(json.dumps(record))
        self.assertEqual(len(h.discover(self.root, None, None, None)), 2)
        self.assertEqual(len(h.discover(self.root, 'codex', 'session', None)), 1)
        (directory / 'broken.json').write_text('{')
        jobs = h.discover(self.root, None, None, None)
        bad = [j for j in jobs if 'discovery_error' in j][0]
        self.assertFalse(h.check(self.root, bad)['ok'])

    def test_regression_and_shrink(self):
        self.set_offset(8)
        previous = self.check()
        self.set_offset(0)
        current = self.check()
        h.check_progress(previous, current)
        self.assertFalse(current['ok'])
        self.transcript.write_bytes(b'{}\n')
        current = self.check()
        h.check_progress(previous, current)
        self.assertTrue(current['ok'])
        self.assertTrue(current['warnings'])

    def test_cli_exit_codes_and_read_only(self):
        before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in (self.cursor, self.transcript)]
        self.assertEqual(self.cli()[0], 0)
        self.assertEqual(self.cli('--require-caught-up')[0], 2)
        after = [(p.read_bytes(), p.stat().st_mtime_ns) for p in (self.cursor, self.transcript)]
        self.assertEqual(before, after)
        self.set_offset(16)
        self.assertEqual(self.cli('--require-caught-up')[0], 0)
        self.set_offset(1)
        self.assertEqual(self.cli()[0], 1)

    def test_wait_timeout_and_stall(self):
        code, reports = self.cli('--wait-seconds', '.03', '--interval-seconds', '.01')
        self.assertEqual(code, 2)
        self.assertEqual(reports[-1]['exit_code'], 2)
        code, reports = self.cli('--wait-seconds', '.1', '--interval-seconds', '.01', '--stall-seconds', '.02')
        self.assertEqual(code, 1)
        self.assertEqual(reports[-1]['captures'][0]['state'], 'stalled')

    def test_wait_freezes_initial_watermark(self):
        real_check = h.check
        calls = 0
        def advance(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                self.transcript.write_bytes(self.transcript.read_bytes() + b'{}\n')
                self.set_offset(16)
            return real_check(*args)
        with patch.object(h, 'check', side_effect=advance):
            code, reports = self.cli('--wait-seconds', '1', '--interval-seconds', '.01')
        self.assertEqual(code, 0)
        self.assertFalse(reports[-1]['all_caught_up'])
        self.assertEqual(reports[-1]['captures'][0]['goal_bytes'], 16)

    def test_checkpoint_fingerprint_and_replacement(self):
        import hashlib
        import os
        data = self.transcript.read_bytes()
        stat = self.transcript.stat()
        anchor = dict(prefix_sha256=hashlib.sha256(data).hexdigest(),
                      boundary_sha256=hashlib.sha256(data).hexdigest())
        if os.name == 'posix':
            anchor.update(device=stat.st_dev, inode=stat.st_ino)
        self.cursor.write_text(json.dumps(dict(offset=len(data), transcript_anchor=anchor)))
        self.assertTrue(self.check()['anchor_verified'])
        self.transcript.write_bytes(data.replace(b'1', b'9'))
        result = self.check()
        self.assertFalse(result['ok'])
        self.assertFalse(result['anchor_verified'])

    def test_no_jobs(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(h.main(['--project', str(self.root)]), 1)

    def test_reject_nonfinite_timing(self):
        for value in ('nan', 'inf', '-1'):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    h.main(['--wait-seconds', value])


if __name__ == '__main__':
    unittest.main()
