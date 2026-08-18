import { store } from "./store.js";
import { api } from "./api.js";
import { withBusy } from "./busy.js";
import { errorDetailHtml } from "./format.js";

import "./components/app-header.js";
import "./components/step-indicator.js";
import "./components/new-job-panel.js";
import "./components/active-panel.js";
import "./components/done-panel.js";
import "./components/failed-panel.js";
import "./components/settings-panel.js";
import "./components/diagnostics-panel.js";
import "./components/manage-panel.js";

function showSetupError(message, detail, info){
  const b = document.getElementById("setupBanner");
  b.innerHTML = errorDetailHtml(message, detail || "");
  b.className = `banner ${info ? "info" : "bad"}`;
  b.classList.remove("hidden");
  b.scrollIntoView({behavior: "smooth", block: "center"});
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
    : job.status === "done"    ? `Finished · ${base}`
    : job.status === "error"   ? `Couldn't be dubbed · ${base}`
    : base;
}

async function refreshDoctor(){
  try{
    const doctor = await api.doctor();
    store.setDoctor(doctor);
    if(!doctor.ready){
      showSetupError("Something needed is missing — see Setup check below.", "", true);
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

function listen(){
  const es = api.events();
  // /api/events replays every known job on connect, and reconnects every few
  // seconds if the stream drops. Only a job that has just *changed* state to
  // done/cancelled is worth a full storage walk.
  es.onmessage = e => {
    const job = JSON.parse(e.data);
    const prevStatus = store.state.jobs[job.id]?.status;
    const settled = prevStatus !== job.status
                  && (job.status === "done" || job.status === "cancelled");
    store.setJob(job);
    const current = store.state.current;
    if(!current || job.id === current || job.status === "running"){
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

document.addEventListener("save-settings", async e => {
  const source = e.target;
  source.setBusy?.(true, "Saving…", "save");
  try{
    const settings = await api.saveSettings(e.detail.data);
    store.setSettings(settings);
    refreshDoctor();
    source.showSaved?.();
  }catch(err){
    source.showSaveError?.(err.message);
  }finally{
    source.setBusy?.(false, null, "save");
  }
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
