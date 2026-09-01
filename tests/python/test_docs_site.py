"""Structural regression checks for the generated documentation site."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


class DocsSiteTests(unittest.TestCase):
    def test_ingestion_flowchart_is_renderable_mermaid(self) -> None:
        page = ROOT / "site" / "ingestion-architecture" / "index.html"
        self.assertTrue(page.exists(), "run `mkdocs build` before this test")

        html = page.read_text(encoding="utf-8")
        self.assertRegex(html, re.compile(r'class=["\']?mermaid(?:["\' >])'))
        self.assertIn("mermaid.min.js", html)
        self.assertIn("javascripts/mermaid.js", html)

        mermaid_config = (ROOT / "docs" / "javascripts" / "mermaid.js").read_text(encoding="utf-8")
        self.assertIn('theme: "base"', mermaid_config)
        self.assertNotIn('theme: "dark"', mermaid_config)

    def test_nested_pages_use_the_rooagi_logo(self) -> None:
        page = ROOT / "site" / "quickstart" / "index.html"
        self.assertTrue(page.exists(), "run `mkdocs build` before this test")

        html = page.read_text(encoding="utf-8")
        self.assertIn("../assets/rooagi-logo.png", html)
        self.assertTrue((ROOT / "site" / "assets" / "rooagi-logo.png").is_file())

    def test_scenario_overlap_stays_inside_the_hero_edges(self) -> None:
        """Structural check: hero and overlap share one grid track."""
        css = (ROOT / "docs" / "stylesheets" / "extra.css").read_text(encoding="utf-8")
        homepage = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")

        self.assertIn('<div class="hero-stack">', homepage)
        self.assertRegex(css, re.compile(r"\.hero-stack\s*\{[^}]*display:\s*grid", re.DOTALL))
        self.assertRegex(css, re.compile(r"\.hero-stack\s*\{[^}]*grid-template-columns:\s*minmax\(0,\s*1fr\)", re.DOTALL))


if __name__ == "__main__":
    unittest.main()
