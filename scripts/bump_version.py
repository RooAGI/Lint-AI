#!/usr/bin/env python3
"""Update Membrick's active package and release metadata to one version.

Usage: python3 scripts/bump_version.py 0.6.0

Historical release notes, benchmark artifacts, and archived outputs are
intentionally left unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CURRENT_CARGO_VERSION = re.search(
    r'^version = "(\d+\.\d+\.\d+)"$',
    (ROOT / "Cargo.toml").read_text(),
    re.MULTILINE,
)
if CURRENT_CARGO_VERSION is None:
    raise RuntimeError("could not read the current version from Cargo.toml")
OLD_VERSION = CURRENT_CARGO_VERSION.group(1)


def replace_marker(path: str, old: str, new: str) -> None:
    file = ROOT / path
    content = file.read_text()
    if old not in content:
        raise ValueError(f"expected version marker not found in {path}: {old!r}")
    file.write_text(content.replace(old, new))


def replace_pattern(path: str, pattern: str, replacement: str) -> None:
    file = ROOT / path
    content = file.read_text()
    updated, count = re.subn(pattern, replacement, content, count=1)
    if count != 1:
        raise ValueError(f"expected one version marker in {path}")
    file.write_text(updated)


def replace_cargo_package_version(path: str, package: str, version: str) -> None:
    file = ROOT / path
    content = file.read_text()
    pattern = re.compile(
        rf'(\[\[package\]\]\nname = "{re.escape(package)}"\nversion = "){re.escape(OLD_VERSION)}(")'
    )
    updated, count = pattern.subn(rf"\g<1>{version}\2", content)
    if count != 1:
        raise ValueError(f"expected one {package} lockfile entry in {path}; found {count}")
    file.write_text(updated)


def replace_json(path: str, version: str, *, node_lock: bool = False) -> None:
    file = ROOT / path
    data = json.loads(file.read_text())
    data["version"] = version
    if node_lock:
        root_package = data["packages"][""]
        root_package["version"] = version
        root_package["optionalDependencies"] = {
            name: version if value == OLD_VERSION else value
            for name, value in root_package.get("optionalDependencies", {}).items()
        }
    elif path.endswith("bindings/node/package.json"):
        data["optionalDependencies"] = {
            name: version if value == OLD_VERSION else value
            for name, value in data.get("optionalDependencies", {}).items()
        }
    file.write_text(json.dumps(data, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="new semantic version, for example 0.6.0")
    args = parser.parse_args()
    if not re.fullmatch(r"\d+\.\d+\.\d+", args.version):
        parser.error("version must use MAJOR.MINOR.PATCH format")
    if not (ROOT / "docs/releases" / f"{args.version}.md").is_file():
        parser.error(f"create docs/releases/{args.version}.md before bumping")

    version = args.version
    if version == OLD_VERSION:
        print(f"Active Membrick version metadata is already {version}.")
        return 0
    for path in ("Cargo.toml", "pyproject.toml", "bindings/node/Cargo.toml"):
        replace_marker(path, f'version = "{OLD_VERSION}"', f'version = "{version}"')
    replace_cargo_package_version("Cargo.lock", "membrick", version)
    replace_cargo_package_version("bindings/node/Cargo.lock", "membrick", version)
    replace_cargo_package_version("bindings/node/Cargo.lock", "membrick-node", version)
    replace_marker("uv.lock", f'name = "membrick"\nversion = "{OLD_VERSION}"',
                   f'name = "membrick"\nversion = "{version}"')

    replace_json("bindings/node/package.json", version)
    replace_json("bindings/node/package-lock.json", version, node_lock=True)
    replace_marker("bindings/node/index.js", OLD_VERSION, version)
    replace_marker("CITATION.cff", f"version: {OLD_VERSION}", f"version: {version}")
    replace_marker("charts/membrick/Chart.yaml", f"version: {OLD_VERSION}", f"version: {version}")
    replace_marker("charts/membrick/Chart.yaml", f'appVersion: "{OLD_VERSION}"', f'appVersion: "{version}"')
    replace_json("src/integrations/openclaw/plugin/package.json", version)
    replace_marker("src/integrations/openclaw/plugin/openclaw.plugin.json",
                   f'"version":"{OLD_VERSION}"', f'"version":"{version}"')
    replace_marker(".github/workflows/python-publish.yml", f"default: v{OLD_VERSION}", f"default: v{version}")
    # The homepage can lag behind Cargo.toml, so update its displayed version
    # independently instead of assuming its current copy matches OLD_VERSION.
    replace_pattern("docs/index.md", r"Version \d+\.\d+\.\d+", f"Version {version}")
    replace_marker("docs/docker.md", f":{OLD_VERSION}", f":{version}")
    replace_marker("docs/docker.md", f"image.tag={OLD_VERSION}", f"image.tag={version}")

    mkdocs = ROOT / "mkdocs.yml"
    nav = mkdocs.read_text()
    entry = f"      - {OLD_VERSION}: releases/{OLD_VERSION}.md"
    new_entry = f"      - {version}: releases/{version}.md"
    if entry not in nav:
        if new_entry in nav:
            print(f"Active Membrick version metadata is already {version}.")
            return 0
        raise ValueError("could not find the current release entry in mkdocs.yml")
    mkdocs.write_text(nav.replace(entry, f"{new_entry}\n{entry}"))
    print(f"Updated active Membrick version metadata to {version}.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"version bump failed: {error}", file=sys.stderr)
        raise SystemExit(1)
