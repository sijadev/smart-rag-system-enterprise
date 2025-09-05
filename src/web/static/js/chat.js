(function(){
    const ids = {
        neo4j: 'sidebar-neo4j-connection',
        qdrant: 'sidebar-db-connection',
        llm: 'sidebar-llm-status'
    };

    const setStatus = (el, ok) => {
        if(!el) return;
        el.textContent = ok ? 'Verbunden' : 'Fehlt';
        el.style.color = ok ? '#10b981' : '#ef4444';
    };

    const showToast = (msg) => {
        let toast = document.getElementById('status-toast');
        if(!toast){
            toast = document.createElement('div');
            toast.id = 'status-toast';
            Object.assign(toast.style, {
                position: 'fixed',
                right: '20px',
                top: '20px',
                background: '#111827',
                color: '#e5e7eb',
                padding: '10px 14px',
                borderRadius: '8px',
                boxShadow: '0 6px 24px rgba(0,0,0,0.5)',
                zIndex: 2000,
                opacity: '0',
                transition: 'opacity 200ms ease-in-out'
            });
            document.body.appendChild(toast);
        }
        toast.textContent = msg;
        requestAnimationFrame(()=>{ toast.style.opacity = '1'; });
        clearTimeout(toast._hide);
        toast._hide = setTimeout(()=>{ toast.style.opacity = '0'; }, 5000);
    };

    const parseDomState = ()=>{
        const out = {};
        Object.entries(ids).forEach(([k,id])=>{
            const el = document.getElementById(id);
            if(!el){ out[k]=false; return; }
            const txt = (el.textContent||'').toLowerCase();
            out[k] = /verbund|connected|ok|online/.test(txt);
        });
        return out;
    };

    const fetchStatus = async (timeoutMs=2500)=>{
        try{
            const controller = new AbortController();
            const id = setTimeout(()=>controller.abort(), timeoutMs);
            const res = await fetch('/api/status', { signal: controller.signal });
            clearTimeout(id);
            if(!res.ok) return null;
            const json = await res.json();
            return { neo4j: !!json.neo4j, qdrant: !!json.qdrant, llm: !!json.llm };
        }catch(e){ return null; }
    };

    document.addEventListener('DOMContentLoaded', async ()=>{
        let known = null;

        // Start check: try API, otherwise derive from DOM once
        const api = await fetchStatus();
        if(api){ known = api; Object.keys(api).forEach(k=> setStatus(document.getElementById(ids[k]), api[k])); }
        else { known = parseDomState(); Object.keys(known).forEach(k=> setStatus(document.getElementById(ids[k]), known[k])); }

        // SSE (EventSource) client: wenn verfügbar, empfängt Status-Änderungen push-basiert
        try{
            if(typeof EventSource !== 'undefined'){
                const es = new EventSource('/api/status/stream');
                es.addEventListener('message', (ev)=>{
                    try{
                        const data = JSON.parse(ev.data);
                        // nur anwenden, wenn sich etwas geändert
                        ['neo4j','qdrant','llm'].forEach(k=>{
                            if(known == null || data[k] !== known[k]){
                                setStatus(document.getElementById(ids[k]), !!data[k]);
                                const human = k === 'qdrant' ? 'Qdrant' : (k === 'neo4j' ? 'Neo4j' : 'LLM');
                                const msg = data[k] ? `${human} verbunden` : `${human} Verbindung verloren`;
                                showToast(msg);
                            }
                        });
                        known = data;
                    }catch(e){ /* ignore parse errors */ }
                });
                es.addEventListener('error', ()=>{
                    // EventSource ggf. nicht erreichbar; schließe und nutze Fallback
                    try{ es.close(); }catch(e){}
                });
            }
        }catch(e){ /* ignore */ }

        // MutationObserver: observe sidebar-status and handle changes
        const container = document.querySelector('.sidebar-status');
        if(!container) return;
        let lastNotify = 0;
        const MIN_INTERVAL = 15000; // 15s Debounce

        const checkAndNotify = async (fromApiIfAvailable=false)=>{
            const now = Date.now();
            if(now - lastNotify < MIN_INTERVAL) return;

            // Prefer structured API if available
            let state = null;
            if(fromApiIfAvailable) state = await fetchStatus();
            if(!state) state = parseDomState();

            ['neo4j','qdrant','llm'].forEach(k=>{
                if(known == null || state[k] !== known[k]){
                    setStatus(document.getElementById(ids[k]), !!state[k]);
                    const human = k === 'qdrant' ? 'Qdrant' : (k === 'neo4j' ? 'Neo4j' : 'LLM');
                    const msg = state[k] ? `${human} verbunden` : `${human} Verbindung verloren`;
                    showToast(msg);
                }
            });
            known = state;
            lastNotify = now;
        };

        const observer = new MutationObserver((mutations)=>{
            // on DOM change, try to get structured API state, but ensure debounce
            checkAndNotify(true).catch(()=>{});
        });
        observer.observe(container, { childList: true, subtree: true, characterData: true });

        // Rare polling fallback to recover from missed events (e.g. 5 minutes)
        setInterval(()=>{ checkAndNotify(true).catch(()=>{}); }, 5*60*1000);

        // Manual retry on Qdrant label
        const retryEl = document.getElementById(ids.qdrant);
        if(retryEl){ retryEl.style.cursor='pointer'; retryEl.title='Klicken zum erneuten Prüfen'; retryEl.addEventListener('click', ()=> checkAndNotify(true)); }
    });
})();

(function(){
    // Import-Button: Datei lesen und an Backend senden
    document.addEventListener('DOMContentLoaded', ()=>{
        try{
            const importBtn = document.getElementById('import-btn');
            const importInput = document.getElementById('import-file');
            const spinner = document.getElementById('import-spinner');
            const result = document.getElementById('import-result');
            if(!importBtn || !importInput) return;

            const setBusy = (busy)=>{
                if(importBtn) importBtn.disabled = busy;
                if(spinner) spinner.style.display = busy ? 'block' : 'none';
            };

            importBtn.addEventListener('click', async (e)=>{
                e.preventDefault();
                result && (result.textContent = '');
                const file = importInput.files && importInput.files[0];
                if(!file){
                    if(result) result.textContent = 'Bitte Datei auswählen.';
                    return;
                }

                try{
                    setBusy(true);
                    const fd = new FormData();
                    fd.append('file', file);
                    // optional extra params: fd.append('source','web');

                    const res = await fetch('/import', { method: 'POST', body: fd });
                    if(!res.ok){
                        const txt = await res.text().catch(()=>null);
                        throw new Error(txt || `HTTP ${res.status}`);
                    }
                    const json = await res.json().catch(()=>null);
                    if(json && json.message) result.textContent = json.message;
                    else result.textContent = 'Import erfolgreich.';
                }catch(err){
                    console.error('Import error', err);
                    result.textContent = 'Import fehlgeschlagen: ' + (err.message || err);
                }finally{
                    setBusy(false);
                }
            });
        }catch(e){ console.error('[import-wiring] error', e); }
    });
})();
