"""Tests for the QDS command-line front-end.

The command line adds no physics of its own, so these tests do not re-check the
engine's numbers -- ``test_detect`` and ``test_attacks`` already do that.  What
they check is that the front-end is a faithful conduit: that every subcommand
dispatches, that ``--json`` emits something ``json.loads`` accepts, that the
honest ideal link comes back clean and accepting, that a forgery comes back
compromised on an authentication failure, that an unknown attack name is a
clean non-zero exit rather than a traceback, and -- the one assertion with
teeth -- that ``selfcheck`` agrees every attack in the suite still triggers
exactly what it was built to trigger.

The whole attack catalogue costs about a second per attack to mount, so it is
mounted exactly once in :func:`setUpModule`, through the same per-seed cache the
``sweep`` and ``selfcheck`` subcommands share.  Every test that needs the sweep
reads from that single mount; the lighter ``run`` and ``attack`` tests spin one
session each, which is cheap enough to leave alone.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import unittest

import qds.attacks as attacks
from qds import cli

_ENGINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_cli(args):
    """Drive :func:`cli.main` in-process, capturing stdout, stderr, and code."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(args))
    return code, out.getvalue(), err.getvalue()


def setUpModule():
    """Mount the whole catalogue once; sweep/selfcheck reuse it via the cache."""
    cli._run_all(0)


# ----------------------------------------------------------------------------
# list
# ----------------------------------------------------------------------------

class TestList(unittest.TestCase):

    def test_human_shows_all_five_families(self):
        code, out, _ = run_cli(["list"])
        self.assertEqual(code, 0)
        self.assertEqual(len(attacks.FAMILIES), 5)
        for family in attacks.FAMILIES:
            self.assertIn(family, out)

    def test_json_covers_every_family_and_variant(self):
        code, out, _ = run_cli(["list", "--json"])
        self.assertEqual(code, 0)
        doc = json.loads(out)
        self.assertEqual(doc["n_families"], 5)
        self.assertEqual({f["family"] for f in doc["families"]},
                         set(attacks.FAMILIES))
        total = sum(len(attacks.CATALOGUE[f]) for f in attacks.FAMILIES)
        self.assertEqual(doc["n_attacks"], total)
        for entry in doc["families"]:
            self.assertEqual(entry["variants"],
                             list(attacks.CATALOGUE[entry["family"]]))


# ----------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------

class TestRun(unittest.TestCase):

    def test_ideal_run_is_clean_and_accepting(self):
        code, out, _ = run_cli(["run", "--noise", "ideal"])
        self.assertEqual(code, 0)
        self.assertIn("CLEAN", out)
        self.assertRegex(out, r"accepting\s+yes")

    def test_ideal_run_json_is_clean_and_accepting(self):
        code, out, _ = run_cli(["run", "--noise", "ideal", "--json"])
        self.assertEqual(code, 0)
        doc = json.loads(out)
        self.assertEqual(doc["verdict"], "clean")
        self.assertTrue(doc["accepting"])
        self.assertEqual(doc["flagged"], [])
        self.assertIn("report", doc)

    def test_intercept_resend_on_ideal_is_compromised(self):
        # The security demonstration: drop a textbook eavesdropper onto an
        # otherwise perfect link and the engine must refuse the signature.
        code, out, _ = run_cli(
            ["run", "--noise", "ideal", "--intervention", "intercept_resend",
             "--json"])
        self.assertEqual(code, 0)
        doc = json.loads(out)
        self.assertEqual(doc["verdict"], "compromised")
        self.assertFalse(doc["accepting"])

    def test_bad_bits_is_a_clean_nonzero_error(self):
        code, out, err = run_cli(["run", "--bits", "7"])
        self.assertNotEqual(code, 0)
        self.assertIn("multiple of 8", err)

    def test_unknown_intervention_is_a_clean_nonzero_error(self):
        code, _, err = run_cli(["run", "--intervention", "no_such_probe"])
        self.assertNotEqual(code, 0)
        self.assertIn("unknown intervention", err)


# ----------------------------------------------------------------------------
# attack
# ----------------------------------------------------------------------------

class TestAttack(unittest.TestCase):

    def test_forgery_single_copy_is_compromised_auth_failure(self):
        code, out, _ = run_cli(["attack", "forgery", "single_copy"])
        self.assertEqual(code, 0)
        self.assertIn("COMPROMISED", out)
        self.assertIn("authentication_failure", out)

    def test_forgery_single_copy_json(self):
        code, out, _ = run_cli(["attack", "forgery", "single_copy", "--json"])
        self.assertEqual(code, 0)
        doc = json.loads(out)
        self.assertEqual(doc["finding"]["verdict"], "compromised")
        self.assertEqual(doc["finding"]["attribution"][0],
                         "authentication_failure")
        self.assertTrue(doc["finding"]["authentication_failed"])
        self.assertTrue(doc["matched"])

    def test_physics_probe_has_no_engine_verdict(self):
        code, out, _ = run_cli(
            ["attack", "forgery", "physics_single_copy", "--json"])
        self.assertEqual(code, 0)
        doc = json.loads(out)
        # a physics-isolation probe carries no engine report, but still reports
        # a (vacuously) matched expectation
        self.assertNotIn("finding", doc)
        self.assertTrue(doc["matched"])

    def test_unknown_family_exits_nonzero(self):
        code, _, err = run_cli(["attack", "no_such_family", "x"])
        self.assertNotEqual(code, 0)
        self.assertIn("unknown attack family", err)

    def test_unknown_variant_exits_nonzero(self):
        code, _, err = run_cli(["attack", "channel", "no_such_variant"])
        self.assertNotEqual(code, 0)
        self.assertIn("variant", err)


# ----------------------------------------------------------------------------
# sweep
# ----------------------------------------------------------------------------

class TestSweep(unittest.TestCase):

    def test_json_emits_the_whole_catalogue(self):
        code, out, _ = run_cli(["sweep", "--json"])
        self.assertEqual(code, 0)
        rows = json.loads(out)
        total = sum(len(attacks.CATALOGUE[f]) for f in attacks.FAMILIES)
        self.assertEqual(len(rows), total)
        self.assertTrue(all("name" in r and "family" in r for r in rows))

    def test_human_runs_and_summarises(self):
        code, out, _ = run_cli(["sweep"])
        self.assertEqual(code, 0)
        self.assertIn("matched their expectation", out)


# ----------------------------------------------------------------------------
# selfcheck -- the self-validating gate
# ----------------------------------------------------------------------------

class TestSelfcheck(unittest.TestCase):

    def test_json_passes_and_every_attack_matched(self):
        code, out, _ = run_cli(["selfcheck", "--json"])
        self.assertEqual(code, 0)
        doc = json.loads(out)
        self.assertTrue(doc["ok"])
        self.assertEqual(doc["n_attacks"], doc["n_matched"])
        self.assertTrue(all(a["matched"] for a in doc["attacks"]))
        self.assertTrue(all(a["discrepancies"] == [] for a in doc["attacks"]))

    def test_human_passes(self):
        code, out, _ = run_cli(["selfcheck"])
        self.assertEqual(code, 0)
        self.assertIn("OK: all", out)


# ----------------------------------------------------------------------------
# bench
# ----------------------------------------------------------------------------

class TestBench(unittest.TestCase):

    def test_json_reports_timing_stats(self):
        code, out, _ = run_cli(["bench", "--repeat", "2", "--json"])
        self.assertEqual(code, 0)
        doc = json.loads(out)
        stats = doc["session_seconds"]
        for key in ("min", "median", "mean", "total"):
            self.assertIn(key, stats)
            self.assertGreaterEqual(stats[key], 0.0)
        self.assertEqual(len(stats["samples"]), 2)
        self.assertIsNone(doc["run_all_seconds"])
        self.assertEqual(doc["repeat"], 2)


# ----------------------------------------------------------------------------
# dispatch and the module entry point
# ----------------------------------------------------------------------------

class TestDispatch(unittest.TestCase):

    def test_no_command_prints_help_and_returns_two(self):
        code, out, _ = run_cli([])
        self.assertEqual(code, 2)
        self.assertIn("usage", out.lower())

    def test_every_subcommand_is_wired_to_a_handler(self):
        parser = cli.build_parser()
        # Each subcommand must set a callable 'func' default -- the dispatch
        # table the CLI relies on.
        actions = [a for a in parser._actions
                   if isinstance(a, argparse._SubParsersAction)]
        self.assertEqual(len(actions), 1)
        for name, subparser in actions[0].choices.items():
            with self.subTest(command=name):
                func = subparser.get_default("func")
                self.assertTrue(callable(func), f"{name} has no handler")


class TestModuleEntryPoint(unittest.TestCase):
    """`python3 -m qds ...` must work, which is what __main__.py provides."""

    def _run(self, *args):
        env = dict(os.environ, PYTHONPATH=_ENGINE_ROOT)
        return subprocess.run([sys.executable, "-m", "qds", *args],
                              cwd=_ENGINE_ROOT, env=env,
                              capture_output=True, text=True)

    def test_list_via_module(self):
        proc = self._run("list")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        for family in attacks.FAMILIES:
            self.assertIn(family, proc.stdout)

    def test_version_via_module(self):
        proc = self._run("--version")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("qds", proc.stdout.lower())


if __name__ == "__main__":
    unittest.main()
