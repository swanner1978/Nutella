"""Verify Python syntax for all source and test files."""

from __future__ import annotations

import ast
import pathlib
import sys


def check_directory(root: pathlib.Path) -> list[str]:
    errors: list[str] = []
    for path in root.rglob("*.py"):
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            errors.append(f"{path}: {exc}")
    return errors


def main() -> int:
    src_errors = check_directory(pathlib.Path("src"))
    test_errors = check_directory(pathlib.Path("tests"))
    errors = src_errors + test_errors
    if errors:
        for err in errors:
            print(err)
        return 1
    src_count = len(list(pathlib.Path("src").rglob("*.py")))
    test_count = len(list(pathlib.Path("tests").rglob("*.py")))
    print(f"Syntax OK: {src_count} src + {test_count} test files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
