#!/usr/bin/env python3
"""Diagnose-Skript: zeigt Factory-Registries vor/nach Bootstrap und versucht, Stores zu instanziieren.

Nutzung: python3 scripts/diagnose_bootstrap.py
"""

import sys
import traceback
from pathlib import Path

from src.factories import DatabaseFactory

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def try_create_vector(name):
    try:
        inst = DatabaseFactory.create_vector_store(
            name, {"collection_name": "diag_test"}
        )
        return type(inst).__name__
    except Exception as e:
        return f"FAILED: {e!r}"


def try_create_graph(name):
    try:
        inst = DatabaseFactory.create_graph_store(name, {})
        return type(inst).__name__
    except Exception as e:
        return f"FAILED: {e!r}"


def main():
    print("== Initial registries ==")
    print(" vector:", list(DatabaseFactory._vector_registry.keys()))
    print(" graph :", list(DatabaseFactory._graph_registry.keys()))

    print("\n== Calling bootstrap.register_all_defaults() ==")
    try:
        from src import bootstrap

        bootstrap.register_all_defaults()
        print("bootstrap.register_all_defaults() succeeded")
    except Exception:
        print("bootstrap.register_all_defaults() raised:")
        traceback.print_exc()

    print("\n== After bootstrap registries ==")
    print(" vector:", list(DatabaseFactory._vector_registry.keys()))
    print(" graph :", list(DatabaseFactory._graph_registry.keys()))

    # Try creating vector stores
    print("\n== Instantiate vector stores ==")
    for name in ["chroma", "faiss", "mock"]:
        print(f'create_vector_store("{name}") ->', try_create_vector(name))

    # Try creating graph stores
    print("\n== Instantiate graph stores ==")
    for name in ["neo4j", "mock"]:
        print(f'create_graph_store("{name}") ->', try_create_graph(name))


if __name__ == "__main__":
    main()
