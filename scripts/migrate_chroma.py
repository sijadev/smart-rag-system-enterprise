#!/usr/bin/env python3
"""Lightweight migration from legacy Chroma SQLite to a new Chroma collection or fallback sqlite.

Usage:
  python3 scripts/migrate_chroma.py --src ./chroma_db/chroma.sqlite3.old --dest ./chroma_db_migrated --collection rag_documents

The script will:
- Inspect the legacy sqlite file, try to extract segment text and embeddings if present.
- Try to initialize a chromadb.Client (current environment). If successful, create/get the target collection and upsert documents (+embeddings when available).
- If chromadb is not usable, write documents into a simple fallback sqlite in the dest directory (chroma_fallback.sqlite3).

This is defensive and logs progress.
"""
import argparse
import json
import os
import sqlite3
import sys
import traceback
from typing import List, Optional, Tuple


def read_legacy(db_path: str) -> Tuple[List[dict], List[Optional[List[float]]]]:
    """Return list of docs dicts and parallel list of embeddings or None."""
    docs = []
    embs = []
    if not os.path.exists(db_path):
        raise FileNotFoundError(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Inspect tables
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]

    # Prefer 'segments' table for document text
    if 'segments' in tables:
        # get columns
        cur.execute("PRAGMA table_info(segments)")
        cols = [r[1] for r in cur.fetchall()]
        # try to find text-like column
        text_col = None
        for candidate in ['text', 'content', 'document', 'chunk', 'segment']:
            if candidate in cols:
                text_col = candidate
                break
        # fallback to first text-like column
        if text_col is None:
            # choose first column that is type TEXT
            cur.execute("PRAGMA table_info(segments)")
            info = cur.fetchall()
            for cid, name, ctype, *_ in info:
                if 'CHAR' in (ctype or '').upper() or 'TEXT' in (ctype or '').upper() or ctype == '':
                    text_col = name
                    break
        if text_col is None:
            # fallback to selecting whole row and joining
            cur.execute("SELECT rowid, * FROM segments LIMIT 100")
            for row in cur.fetchall():
                docs.append({'content': json.dumps(row)})
                embs.append(None)
        else:
            cur.execute(f"SELECT rowid, {text_col} FROM segments")
            for row in cur.fetchall():
                rid, text = row[0], row[1]
                docs.append({'id': f'segment_{rid}', 'content': text or ''})
                embs.append(None)
    else:
        # try collections/segments alternative
        if 'collections' in tables and 'segments' not in tables:
            # nothing useful, bail
            pass

    # Try to read embeddings table if present
    if 'embeddings' in tables:
        # Attempt to read mapping between segment/document id and embedding vector
        # The schema varies; we'll try common variants
        cur.execute("PRAGMA table_info(embeddings)")
        info = cur.fetchall()
        cols = [r[1] for r in info]
        # Possible columns: id, embedding, embedding_data, vector, segment_id
        # We'll try to read all embeddings into a dict by id
        try:
            cur.execute("SELECT * FROM embeddings LIMIT 0")
            has_rows = True
        except Exception:
            has_rows = False
        emb_map = {}
        if has_rows:
            cur.execute("SELECT * FROM embeddings")
            rows = cur.fetchall()
            # map by first column (id) to blob/text
            for r in rows:
                key = str(r[0])
                val = None
                # try to find a numeric vector representation in row
                for cell in r[1:]:
                    if isinstance(cell, str):
                        try:
                            # maybe JSON
                            cand = json.loads(cell)
                            if isinstance(cand, list):
                                val = [float(x) for x in cand]
                                break
                        except Exception:
                            pass
                    if isinstance(cell, (bytes, memoryview)):
                        # try to decode numpy save? skip for safety
                        pass
                    if isinstance(cell, (list, tuple)):
                        val = [float(x) for x in cell]
                        break
                if val is not None:
                    emb_map[key] = val
        # if we have embeddings, attach to docs by id if possible
        if emb_map:
            for i, d in enumerate(docs):
                key = d.get('id')
                if key and key in emb_map:
                    embs[i] = emb_map[key]

    conn.close()
    return docs, embs


def write_to_chroma_client(docs: List[dict], embs: List[Optional[List[float]]], dest_dir: str, collection_name: str) -> bool:
    try:
        import chromadb
        try:
            from chromadb.config import Settings
        except Exception:
            Settings = chromadb.config.Settings
        settings = Settings(anonymized_telemetry=False, allow_reset=False, chroma_db_impl='sqlite', persist_directory=dest_dir)
        client = chromadb.Client(settings)
        try:
            collection = client.get_collection(collection_name)
        except Exception:
            collection = client.create_collection(name=collection_name, metadata={'hnsw:space': 'cosine'})
        docs_texts = [d.get('content', '') for d in docs]
        metas = [{k: v for k, v in d.items() if k != 'content'} for d in docs]
        # prepare embeddings list with None replaced by zeros if necessary
        emb_list = []
        for e in embs:
            if e is None:
                emb_list.append([0.0] * 384)
            else:
                emb_list.append(e)
        ids = [d.get('id', f'doc_{i}') for i, d in enumerate(docs)]
        collection.add(ids=ids, documents=docs_texts, metadatas=metas, embeddings=emb_list)
        return True
    except Exception:
        traceback.print_exc()
        return False


def write_to_fallback_sqlite(docs: List[dict], embs: List[Optional[List[float]]], dest_dir: str) -> bool:
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, 'chroma_fallback.sqlite3')
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, content TEXT, metadata TEXT, embedding TEXT)''')
    for i, d in enumerate(docs):
        _id = d.get('id', f'doc_{i}')
        cont = d.get('content', '')
        meta = json.dumps({k: v for k, v in d.items() if k != 'content' and k != 'id'})
        emb = embs[i]
        emb_json = json.dumps(emb) if emb is not None else None
        cur.execute('INSERT OR REPLACE INTO documents (id, content, metadata, embedding) VALUES (?, ?, ?, ?)', (_id, cont, meta, emb_json))
    conn.commit()
    conn.close()
    return True


def main(argv: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--dest', required=True)
    parser.add_argument('--collection', default='rag_documents')
    args = parser.parse_args(argv)

    try:
        docs, embs = read_legacy(args.src)
        print(f'Read {len(docs)} docs from legacy DB')
    except Exception as e:
        print('Failed to read legacy DB:', e)
        traceback.print_exc()
        sys.exit(2)

    # try to write to chroma client
    ok = write_to_chroma_client(docs, embs, args.dest, args.collection)
    if ok:
        print('Successfully wrote to chroma client in', args.dest)
        sys.exit(0)

    print('Writing to fallback sqlite in', args.dest)
    ok2 = write_to_fallback_sqlite(docs, embs, args.dest)
    if ok2:
        print('Successfully wrote fallback sqlite in', args.dest)
        sys.exit(0)
    print('Migration failed')
    sys.exit(3)
