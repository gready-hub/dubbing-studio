import { store } from "./store.js";
import { api } from "./api.js";
import { withBusy } from "./busy.js";
import { errorDetailHtml } from "./format.js";

import "./components/app-header.js";
import "./components/new-job-panel.js";
import "./components/active-panel.js";
import "./components/done-panel.js";
import "./components/failed-panel.js";
import "./components/settings-panel.js";
import "./components/diagnostics-panel.js";
import "./components/manage-panel.js";

const stillMotion = matchMedia("(prefers-reduced-motion: reduce)");

function showSetupError(message, detail, info){
  const b = document.getElementById("setupBanner");
  b.innerHTML = errorDetailHtml(message, detail || "");
  b.className = `banner ${info ? "info" : "bad"}`;
  b.classList.remove("hidden");
  b.scrollIntoView({behavior: stillMotion.matches ? "auto" : "smooth", block: "center"});
}

function hideSetupError(){
  document.getElementById("setupBanner").classList.add("hidden");
}

// A 10-minute video took 48 minutes to dub on this machine, so the app is
// routinely left running behind something else. The window title is the one
// thing still visible from another application — in the Dock, in the window
// list, on a browser tab — and it never changed.
function setWindowTitle(job){
  const base = "Dubbing Studio";
  const at = job && Math.round((job.overall || 0) * 100);
  document.title =
    !job || job.status === "queued" ? base
    : job.status === "running" ? `${at}% · ${base}`
    : job.status === "done"      ? `Done · ${base}`
    : job.status === "error"     ? `Failed · ${base}`
    : job.status === "cancelled" ? `Cancelled · ${base}`
    : base;
}

async function refreshDoctor(){
  try{
    const doctor = await api.doctor();
    store.setDoctor(doctor);
    if(!doctor.ready){
      showSetupError("Something needed is missing. See Setup check below.", "", true);
    }
  }catch(err){ /* best effort */ }
}

async function refreshStorage(){
  try{
    store.setStorage(await api.storage());
  }catch(err){
    document.querySelector("manage-panel")?.showStorageCheckError();
  }
}

// A finished sample lives only in the view — it has no list of its own the
// way a dubbed video or a failure does — so it is shown again only while it
// is the most recent thing that happened, of any kind: another sample, a
// full run, even one that failed. Anything newer means the user has moved
// on to that instead.
function latestSample(jobs){
  const all = Object.values(jobs);
  if(!all.length) return null;
  const activity = j => j.finished || j.started || 0;
  const newest = all.reduce((a, b) => activity(b) > activity(a) ? b : a);
  return (newest.status === "done" && newest.preview) ? newest.id : null;
}

function listen(){
  const es = api.events();
  // /api/events replays every known job on connect, and reconnects every few
  // seconds if the stream drops. Only a job that has just *changed* state to
  // done, cancelled or errored is worth a full storage walk: whatever it
  // downloaded or transcribed on the way there is still sitting in its
  // workdir whichever of the three it ended on, and the storage panel's
  // per-job breakdown goes stale after any of them, not just the first.
  es.onmessage = e => {
    const job = JSON.parse(e.data);
    const prevStatus = store.state.jobs[job.id]?.status;
    const settled = prevStatus !== job.status
                  && (job.status === "done" || job.status === "cancelled"
                      || job.status === "error");
    store.setJob(job);
    // /api/events replays every known job oldest-first on connect, so a job
    // that has already finished must never take the view: only a running one
    // does that, and a queued one only when nothing holds it yet.
    const current = store.state.current;
    if(job.status === "running" || job.id === current
       || (!current && job.status === "queued")){
      store.setCurrent(job.id);
    }
    if(job.id === store.state.current) setWindowTitle(job);
    // The banner at the top is for problems with the submission itself. A job
    // that got as far as running and then failed belongs where the job was.
    if(job.status === "error") hideSetupError();
    if(settled) refreshStorage();
  };
  es.onerror = () => setTimeout(()=>{ es.close(); listen(); }, 4000);
}

async function checkVersion(){
  try{
    const v = await api.version();
    if(!v.known || !v.update) return;
    document.querySelector("app-header")?.showUpdate(v);
  }catch(err){ /* never worth bothering anyone about */ }
}

async function boot(){
  const initial = await api.state();
  store.set({
    settings: initial.settings,
    settings_defaults: initial.settings_defaults,
    settings_path: initial.settings_path,
    features: initial.features,
    presets: initial.presets,
    glossaries: initial.glossaries,
    voices: initial.voices,
    machine: initial.machine,
    output_dir: initial.output_dir,
  });

  const jobs = {};
  initial.jobs.forEach(j => { jobs[j.id] = j; });
  store.set({jobs});
  const live = initial.jobs.find(j => j.status === "running" || j.status === "queued");
  if(live) store.setCurrent(live.id);
  else {
    const sample = latestSample(jobs);
    if(sample) store.setCurrent(sample);
  }

  await refreshDoctor();
  await refreshStorage();
  listen();
  checkVersion();
}

document.addEventListener("start-job", async e => {
  const { url, preview } = e.detail;
  const panel = document.querySelector("new-job-panel");
  panel.setUrl(url);
  hideSetupError();
  panel.setBusy(true, null, "start");
  try{
    const job = await api.startJob(url, preview);
    store.setJob(job);
    // Don't steal the view from a job that is actually running — the new one
    // is queued behind it and shows up in the waiting list instead.
    if(!Object.values(store.state.jobs).some(j => j.status === "running")){
      store.setCurrent(job.id);
    }
  }catch(err){
    showSetupError(err.message, err.detail);
  }finally{
    panel.setBusy(false, null, "start");
  }
});

document.addEventListener("cancel-job", e => {
  api.cancelJob(e.detail.id).catch(()=>{});
});

// A settings write ends in the same two places whichever way it was made: the
// reply is what the store holds from then on, and the setup check is re-run,
// because an API key or a model named in the settings decides what passes it.
async function writeSettings(source, {label, busy, write, done}){
  source.setBusy?.(true, label, busy);
  try{
    store.setSettings(await write());
    refreshDoctor();
    done();
  }catch(err){
    source.showSaveError?.(err.message);
  }finally{
    source.setBusy?.(false, null, busy);
  }
}

document.addEventListener("save-settings", e => {
  const source = e.target;
  // A panel whose fields write straight through has no Save button to put the
  // label on, so it is told separately that a save is in flight.
  source.showSaving?.();
  writeSettings(source, {
    label: "Saving…", busy: "save",
    write: () => api.saveSettings(e.detail.data),
    done: () => source.showSaved?.(),
  });
});

document.addEventListener("reset-settings", e => {
  const source = e.target;
  writeSettings(source, {
    // The panel names the button that was pressed; without one, every button in
    // the group.
    label: "Resetting…", busy: e.detail.btn || "reset",
    write: () => api.resetSettings(e.detail.keys),
    done: () => source.showReset?.(),
  });
});

document.addEventListener("reset", () => {
  setWindowTitle(null);
  store.setCurrent(null);
  const panel = document.querySelector("new-job-panel");
  panel.setUrl("");
  panel.focusUrl();
});

document.addEventListener("reveal", e => {
  api.reveal(e.detail.path).catch(()=>{});
});

document.addEventListener("uninstall", async () => {
  try{ await api.uninstall(); }
  catch(err){ showSetupError(err.message, err.detail); }
});

document.addEventListener("clear-storage", async e => {
  const { what, btn } = e.detail;
  await withBusy(btn, "Clearing…", async () => {
    try{
      store.setStorage(await api.clearStorage(what));
    }catch(err){
      showSetupError(err.message, err.detail);
    }
  });
});

document.addEventListener("toggle-awake", async e => {
  try{
    store.setSettings(await api.saveSettings({keep_awake: e.detail.on}));
  }catch(err){
    store.notify();
    showSetupError(err.message, err.detail);
  }
});

document.addEventListener("run-update", async () => {
  try{ await api.runUpdate(); }
  catch(err){ showSetupError(err.message, err.detail); }
});

boot();
