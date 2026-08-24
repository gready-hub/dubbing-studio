async function request(path, opts){
  const r = await fetch(path, opts);
  if(!r.ok){
    const body = await r.json().catch(()=>({}));
    const err = new Error(body.detail || "The request failed.");
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
  // null means every setting the app ships with a default for, which the
  // server reads as "all but the API keys".
  resetSettings: keys => request("/api/settings/reset", json({keys: keys ?? null})),
  startJob: (url, preview) => request("/api/job", json({url, preview: !!preview})),
  cancelJob: id => request(`/api/job/${encodeURIComponent(id)}/cancel`, {method: "POST"}),
  storage: () => request("/api/storage"),
  clearStorage: what => request("/api/storage/clear", json({what})),
  reveal: path => request("/api/reveal", json({path})),
  doctor: () => request("/api/doctor"),
  version: () => request("/api/version"),
  runUpdate: () => request("/api/update", {method: "POST"}),
  // Just yt-dlp, in place, without the full reinstall runUpdate() triggers.
  // It is the dependency that goes stale on YouTube's schedule rather than
  // ours, and the fix has to be cheaper than the failure it prevents.
  updateYtdlp: () => request("/api/ytdlp/update", {method: "POST"}),
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
      throw new Error(detail || "The voice preview failed.");
    }
    return r.blob();
  },

  events(){
    return new EventSource("/api/events");
  },
};
