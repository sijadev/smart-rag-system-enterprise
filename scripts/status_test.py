"""
Minimaler Test-Client für den SSE-Stream und optionales Polling des /api/status Endpoints.

Benutzung:
  pip install requests sseclient-py
  python scripts/status_test.py --stream-url http://127.0.0.1:8000/api/status/stream --status-url http://127.0.0.1:8000/api/status --poll-interval 5

Der Client startet zwei Threads:
- SSE-Listener: empfängt Events vom Stream und gibt die JSON-Payload aus.
- Poller: fragt periodisch /api/status und gibt Änderungen aus (nützlich, um zu sehen ob der Poll-Wert wirkt).
"""

import argparse
import json
import sys
import threading
import time

import requests

try:
    from sseclient import SSEClient
except Exception:
    print("Fehler: benötigtes Paket 'sseclient' nicht gefunden. Installiere mit: pip install sseclient-py")
    sys.exit(1)

# Stop-Event, damit Threads sauber beendet werden können
stop_event = threading.Event()


def sse_listener(stream_url: str):
    print(f"SSE-Listener verbindet zu: {stream_url}")
    backoff = 1
    while not stop_event.is_set():
        resp = None
        try:
            # Für Streaming keine Read-Timeout setzen (None) -> timeout=(connect, read)
            resp = requests.get(stream_url, stream=True, timeout=(5, None))
            # instantiate SSEClient with common fallbacks
            try:
                client = SSEClient(resp)
            except TypeError:
                try:
                    client = SSEClient(resp.raw)
                except Exception:
                    client = SSEClient(resp.iter_lines())

            try:
                iterator = iter(client)
            except TypeError:
                if hasattr(client, 'events'):
                    iterator = client.events()
                else:
                    iterator = client

            # Reset backoff after successful connect
            backoff = 1

            for event in iterator:
                if stop_event.is_set():
                    break
                ts = time.strftime('%Y-%m-%d %H:%M:%S')
                data = None
                if hasattr(event, 'data'):
                    data = event.data
                else:
                    try:
                        if isinstance(event, bytes):
                            text = event.decode('utf-8')
                        else:
                            text = str(event)
                        lines = [ln for ln in text.splitlines() if ln.strip()]
                        data_lines = [ln.partition('data: ')[2] if ln.startswith('data:') else ln for ln in lines]
                        data = '\n'.join(data_lines).strip()
                    except Exception:
                        data = str(event)

                try:
                    parsed = json.loads(data)
                    pretty = json.dumps(parsed, ensure_ascii=False)
                except Exception:
                    pretty = data
                print(f"[{ts}] SSE event: {pretty}")

        except requests.exceptions.ReadTimeout:
            print("SSE: ReadTimeout, reconnecting...")
        except requests.exceptions.ConnectionError as e:
            print(f"SSE: Connection error: {e}. reconnecting in {backoff}s...")
        except Exception as e:
            print(f"SSE: Unerwarteter Fehler: {e}")
        finally:
            try:
                if resp is not None:
                    resp.close()
            except Exception:
                pass

        # exponential backoff with cap
        if stop_event.is_set():
            break
        time.sleep(backoff)
        backoff = min(backoff * 2, 30)


def poll_status(url: str, interval: int):
    print(f"Poller fragt '{url}' alle {interval}s ab")
    last = None
    while not stop_event.is_set():
        try:
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            cur = r.json()
        except Exception as e:
            cur = {"error": str(e)}
        if last is None or cur != last:
            ts = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{ts}] Poll result: {json.dumps(cur, ensure_ascii=False)}")
            last = cur
        time.sleep(interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SSE + Status poll test client')
    parser.add_argument('--stream-url', default='http://127.0.0.1:8000/api/status/stream')
    parser.add_argument('--status-url', default='http://127.0.0.1:8000/api/status')
    parser.add_argument('--poll-interval', type=int, default=5)
    parser.add_argument('--no-poll', action='store_true')
    args = parser.parse_args()

    t1 = threading.Thread(target=sse_listener, args=(args.stream_url,), daemon=True)
    t1.start()

    if not args.no_poll:
        t2 = threading.Thread(target=poll_status, args=(args.status_url, args.poll_interval), daemon=True)
        t2.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print('\nBeendet')
        stop_event.set()
        sys.exit(0)
