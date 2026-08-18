import { BaseElement } from "../base-element.js";
import { store } from "../store.js";
import { escapeHtml, fmt } from "../format.js";
import { STAGE_LABELS } from "../stages.js";

const SHELL = `
  <div class="panel">
    <p class="job-title" id="dTitle"></p>
    <p class="msg" id="dPath"></p>
    <div id="dNotes"></div>
    <video id="dVideo" controls preload="none"></video>
    <p class="hint hidden" id="dGone">That sample has been cleared away — samples live in
      the working files, which get tidied up. Run it again to make a new one.</p>
    <div style="display:flex;gap:8px;margin:16px 0 4px;flex-wrap:wrap" id="dActions"></div>
    <details style="margin-top:14px">
      <summary>Quality report</summary>
      <div id="dStats" style="margin-top:10px"></div>
    </details>
  </div>
`;

class DonePanel extends BaseElement {
  connectedCallback(){
    this.html(SHELL);
    this._unsub = store.subscribe(s => this.update(s));
  }

  disconnectedCallback(){
    this._unsub?.();
  }

  update(s){
    const job = s.jobs[s.current];
    const isDone = !!job && job.status === "done";
    this.hidden = !isDone;
    if(!isDone) return;

    const sample = !!job.preview;
    this.$("#dTitle").textContent = (sample ? "Sample — " : "") + job.title;
    this.$("#dPath").textContent = job.message;

    const vid = this.$("#dVideo");
    if(vid.dataset.job !== job.id){
      vid.dataset.job = job.id;
      // A sample lives in the working files, which Clear working files can take
      // away while this panel is still on screen. Say so instead of leaving a
      // broken player the user has no way to interpret.
      vid.onerror = () => {
        vid.classList.add("hidden");
        this.$("#dGone").classList.remove("hidden");
      };
      vid.classList.remove("hidden");
      this.$("#dGone").classList.add("hidden");
      vid.src = `/api/job/${job.id}/video`;
    }

    const actions = this.$("#dActions");
    if(actions.dataset.job !== job.id){
      actions.dataset.job = job.id;
      actions.innerHTML = sample
        ? `<button class="primary" id="dEscalate">Dub it</button>
           <button class="ghost" id="dReset">Dub another</button>`
        : `<button class="primary" id="dReveal">Show in Finder</button>
           <button class="ghost" id="dReset">Dub another</button>`;
      // Two clicks are one job — the id a full run gets is the same either
      // time, so this reuses the start-job path rather than a separate one.
      const escalate = this.$("#dEscalate");
      if(escalate) escalate.onclick = () => this.emit("start-job", {url: job.url, preview: false});
      const reveal = this.$("#dReveal");
      if(reveal) reveal.onclick = () => this.emit("reveal", {path: job.output});
      this.$("#dReset").onclick = () => this.emit("reset");
    }

    const stats = job.stats || {};
    const yn = v => v === true ? "yes" : v === false ? "no" : "—";
    this.$("#dNotes").innerHTML = (stats.notes || []).length
      ? `<div class="banner info" style="margin:0 0 12px">`
        + (stats.notes || []).map(n=>`<div>${escapeHtml(n)}</div>`).join("") + `</div>`
      : "";
    this.$("#dStats").innerHTML = [
      ...(sample ? [["Sample taken from", stats.preview_from ?? "0:00"]] : []),
      ["Quality preset", stats.preset ?? "—"],
      ["Speakers found", stats.speakers ?? "—"],
      ["Translated by", stats.translated_by ?? "—"],
      ["Voices used", stats.voices ?? "—"],
      ["Music and effects kept", yn(stats.music_kept)],
      ["Lines spoken", stats.lines_spoken ?? "—"],
      ["Needed compressing", `${stats.compressed ?? 0} of ${stats.lines ?? 0}`],
      ["Hardest squeeze", stats.max_factor ? stats.max_factor.toFixed(2)+"x" : "none"],
      ["Timing drift", (stats.max_drift ?? 0)+"s"],
      ["Video frames preserved", stats.frames_match ? `yes (${stats.output_frames})` : "re-encoded"],
      ["Audio vs video length", (stats.drift_seconds ?? 0)+"s apart"],
      ["Dubbed audio level", stats.peak_db == null ? "—"
        : `peak ${stats.peak_db} dB, average ${stats.mean_db} dB`],
      ["Silent stretches", stats.silent_seconds == null ? "—"
        : `${stats.silent_seconds}s (${Math.round((stats.silent_share ?? 0)*100)}% of the `
          + `${sample ? "sample" : "video"})`],
      ["Engine", stats.engine ?? "—"],
      ...(sample ? [] : [["Working files freed",
                          stats.working_files_freed ? stats.working_files_freed+" MB" : "—"]]),
      ["Took", fmt(job.elapsed)]
    ].map(([k,v])=>`<div class="stat"><span>${k}</span><span>${v}</span></div>`).join("")
      // Longest first, and only the stages that took real time.
      + Object.entries(job.stage_times || {})
        .filter(([,sec]) => sec >= 1)
        .sort((a,b) => b[1]-a[1])
        .map(([key,sec]) => `<div class="stat sub"><span>${
          escapeHtml(STAGE_LABELS[key] || key)}</span><span>${fmt(sec)}</span></div>`)
        .join("");
  }
}

customElements.define("done-panel", DonePanel);
