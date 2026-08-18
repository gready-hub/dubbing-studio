async function request(path, opts){
  const r = await fetch(path, opts);
  if(!r.ok){
    const body = await r.json().catch(()=>({}));
    const err = new Error(body.detail || "Something went wrong");
    err.detail = body.detail || "";
    throw err;
  }
  return r.json();
}

const json = data => ({
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify(data),
});

export const api = {
  state: () => request("/api/state"),
  saveSettings: data => request("/api/settings", json({data})),
  startJob: (url, preview) => request("/api/job", json({url, preview: !!preview})),
  cancelJob: id => request(`/api/job/${encodeURIComponent(id)}/cancel`, {method: "POST"}),
  storage: () => request("/api/storage"),
  clearStorage: what => request("/api/storage/clear", json({what})),
  reveal: path => request("/api/reveal", json({path})),
  doctor: () => request("/api/doctor"),
  version: () => request("/api/version"),
  runUpdate: () => request("/api/update", {method: "POST"}),
  uninstall: () => request("/api/uninstall", {method: "POST"}),

  async diagnostics(){
    const r = await fetch("/api/diagnostics");
    if(!r.ok) throw new Error(String(r.status));
    return r.json();
  },

  async voicePreview(voice, speed){
    const r = await fetch(`/api/voice-preview?voice=${encodeURIComponent(voice)}`
                        + `&speed=${encodeURIComponent(speed)}`);
    if(!r.ok){
      const detail = (await r.json().catch(()=>({}))).detail;
      throw new Error(detail || "That voice couldn't be played.");
    }
    return r.blob();
  },

  events(){
    return new EventSource("/api/events");
  },
};
