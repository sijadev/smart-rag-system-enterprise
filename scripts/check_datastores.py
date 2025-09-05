#!/usr/bin/env python3
"""Kurzes Prüfskript für ChromaDB (lokale Persistenz) und Neo4j.

Gebrauch: python3 scripts/check_datastores.py
Optional: --chroma-dir DIR um ein anderes Chroma persist_directory zu prüfen.
Gibt JSON mit Status/Counts auf stdout zurück.
"""
import json
import os
import sys
import traceback
from typing import Any, Dict, List


def check_chroma(persist_dir: str | None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        import chromadb
        from chromadb.config import Settings
    except Exception as e:
        out["error"] = f"chromadb import failed: {e!r}"
        return out

    # Try sqlite impl (commonly used) and duckdb+parquet as fallback
    candidates = [] if persist_dir is None else [persist_dir]
    candidates += ["./chroma_db", os.path.expanduser("~/.chromadb")] + [None]

    for d in candidates:
        try:
            if d is None:
                settings = Settings(anonymized_telemetry=False, allow_reset=True)
                desc = "DEFAULT"
            else:
                settings = Settings(
                    anonymized_telemetry=False,
                    allow_reset=False,
                    chroma_db_impl="sqlite",
                    persist_directory=d,
                )
                desc = f"sqlite:{d}"

            client = chromadb.Client(settings)
            cols = client.list_collections()
            names = [c.get("name") for c in cols]
            out["used_setting"] = desc
            out["collections"] = names
            if "rag_documents" in names:
                col = client.get_collection("rag_documents")
            elif "documents" in names:
                col = client.get_collection("documents")
            elif names:
                col = client.get_collection(names[0])
            else:
                col = None

            if col is not None:
                data = col.get()
                ids = data.get("ids", [[]])
                count = len(ids[0]) if ids and isinstance(ids[0], list) else 0
                out["found_collection"] = col.name if hasattr(col, "name") else "<collection>"
                out["count"] = count
            else:
                out["found_collection"] = None
                out["count"] = 0

            return out
        except Exception as e:
            out.setdefault("tried", []).append({"setting": desc if 'desc' in locals() else str(d), "error": str(e)})
            # continue to next candidate
    return out


def check_chroma_sqlite_file(path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"path": path}
    try:
        import sqlite3
        if not os.path.exists(path):
            out["exists"] = False
            return out
        out["exists"] = True
        conn = sqlite3.connect(path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        out["tables"] = [r[0] for r in cur.fetchall()]
        try:
            cur.execute("SELECT id, name FROM collections")
            out["collections"] = [{"id": r[0], "name": r[1]} for r in cur.fetchall()]
        except Exception as e:
            out["collections_error"] = str(e)
        try:
            cur.execute("SELECT count(*) FROM segments")
            out["segments_count"] = cur.fetchone()[0]
        except Exception as e:
            out["segments_count_error"] = str(e)
        try:
            cur.execute("SELECT count(*) FROM embeddings")
            out["embeddings_count"] = cur.fetchone()[0]
        except Exception as e:
            out["embeddings_count_error"] = str(e)
        conn.close()
    except Exception as e:
        out["error"] = str(e)
        out["trace"] = traceback.format_exc()
    return out


def check_neo4j() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        from neo4j import GraphDatabase
    except Exception as e:
        out["error"] = f"neo4j driver import failed: {e!r}"
        return out

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "neo4j123")
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            total_nodes = session.run("MATCH (n) RETURN count(n) AS c").single().value()
            total_rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single().value()
            try:
                doc_nodes = session.run("MATCH (d:Document) RETURN count(d) AS c").single().value()
            except Exception:
                doc_nodes = None
        out["uri"] = uri
        out["total_nodes"] = int(total_nodes) if total_nodes is not None else None
        out["total_relationships"] = int(total_rels) if total_rels is not None else None
        out["document_nodes"] = int(doc_nodes) if doc_nodes is not None else None
    except Exception as e:
        out["error"] = str(e)
        out["trace"] = traceback.format_exc()
    return out


def main(argv: List[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Check ChromaDB and Neo4j status and counts")
    parser.add_argument("--chroma-dir", help="Chroma persist_directory to try", default=None)
    parser.add_argument("--chroma-sqlite", help="Path to chroma sqlite file to inspect", default="chroma_db/chroma.sqlite3")
    args = parser.parse_args(argv)

    report: Dict[str, Any] = {}
    report["chroma"] = check_chroma(args.chroma_dir)
    report["chroma_sqlite_file"] = check_chroma_sqlite_file(args.chroma_sqlite)
    report["neo4j"] = check_neo4j()

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
