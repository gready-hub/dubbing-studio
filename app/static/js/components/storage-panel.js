import { BaseElement } from "../base-element.js";
import { store } from "../store.js";
import { fmtBytes, escapeAttr } from "../format.js";
import "./list-row.js";

// Warm, distinguishable, and quiet enough to sit in a panel that is not the
// point of the app. Keyed by group so a colour always means the same thing.
const SPACE_COLOURS = {
  jobs:"#b45309", models:"#d97706", hfmodels:"#f59e0b", previews:"#fcd34d",
  venv:"#78716c", ollama:"#a16207", output:"#15803d"
};

const SHELL = `
  <div class="panel">
    <b id="storageSummary" style="font-size:15px"></b>
    <p class="usage-empty" id="usageTotal"></p>
    <div class="usage" id="usage"></div>
    <div id="storageGroups"></div>
    <details id="jobsWrap" class="hidden" style="margin-top:12px">
      <summary id="jobsSummary">Working files, video by video</summary>
      <div id="storageJobs" style="margin-top:6px"></div>
    </details>
    <div class="danger-row">
      <div style="min-width:0">
        <b>Remove Dubbing Studio</b>
        <small>Deletes the app, its models and its working files. Your dubbed
          videos are kept, and anything the rest of your Mac shares — Homebrew,
          ffmpeg, Ollama — is listed rather than removed.</small>
      </div>
      <button class="danger" id="uninstallBtn">Uninstall…</button>
    </div>
  </div>
`;

class StoragePanel extends BaseElement {
  connectedCallback(){
    this.html(SHELL);
    this.$("#uninstallBtn").onclick = () => {
      if(!confirm("Open the uninstaller?\n\nIt shows exactly what it will delete and "
                + "asks again before removing anything. Your dubbed videos are kept."))
        return;
      this.emit("uninstall");
    };
    this._unsub = store.subscribe(s => this.update(s));
  }

  disconnectedCallback(){
    this._unsub?.();
  }

  update(s){
    if(s.storage) this.paint(s.storage);
  }

  showCheckError(){
    this.$("#storageSummary").textContent = "Disk space — couldn't check";
  }

  // One glance answers the question people actually open this panel with —
  // which of these is the big one. The rows below double as its legend.
  paint(s){
    const free = fmtBytes(s.free || 0);
    this.$("#storageSummary").textContent = s.low
      ? `Disk space — only ${free} left`
      : `Disk space — ${free} free`;
    this.$("#storageSummary").style.color = s.low ? "var(--bad)" : "";

    const groups = s.groups || [];
    const total = groups.reduce((n, g) => n + (g.bytes || 0), 0);
    this.$("#usage").innerHTML = groups.filter(g => g.bytes > 0).map(g =>
      `<i style="width:${(g.bytes / total * 100).toFixed(2)}%;background:${SPACE_COLOURS[g.key]
        || "var(--muted)"}" title="${escapeAttr(g.label)} — ${fmtBytes(g.bytes)}"></i>`).join("");
    this.$("#usage").classList.toggle("hidden", !total);
    this.$("#usageTotal").textContent = total
      ? `${fmtBytes(total)} in total` : "Nothing stored yet.";

    const groupsBox = this.$("#storageGroups");
    groupsBox.innerHTML = "";
    groups.forEach(g => {
      const row = document.createElement("list-row");
      row.data = {
        color: SPACE_COLOURS[g.key] || "var(--muted)",
        title: g.label,
        qty: fmtBytes(g.bytes),
        subtitle: g.hint,
        actions: [
          ...(g.path ? [{label: "Show", onClick: () => this.emit("reveal", {path: g.path})}] : []),
          ...(g.clearable ? [{label: "Clear", className: "danger", disabled: !g.bytes,
              onClick: btn => this.clear(g.key, g.label, btn)}] : []),
        ],
      };
      groupsBox.appendChild(row);
    });

    const jobs = s.jobs || [];
    this.$("#jobsWrap").classList.toggle("hidden", !jobs.length);
    this.$("#jobsSummary").textContent = `Working files, video by video (${jobs.length})`;
    const jobsBox = this.$("#storageJobs");
    jobsBox.innerHTML = "";
    jobs.forEach(j => {
      const row = document.createElement("list-row");
      row.data = {
        title: j.title,
        qty: fmtBytes(j.bytes),
        actions: [{label: "Clear", className: "danger",
          onClick: btn => this.clear(`job:${j.id}`, j.title, btn)}],
      };
      jobsBox.appendChild(row);
    });
  }

  clear(what, label, btn){
    if(!confirm(`Delete "${label}"?\n\nFinished videos are never touched. `
              + `Anything removed here is rebuilt or downloaded again when it is next needed.`))
      return;
    this.emit("clear-storage", {what, label, btn});
  }
}

customElements.define("storage-panel", StoragePanel);
