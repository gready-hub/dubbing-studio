import { BaseElement } from "../base-element.js";
import { store } from "../store.js";
import { escapeHtml, fmt } from "../format.js";
import { OUTPUT_GONE, SAMPLE_GONE, knownGone, outputGone, runRows, statRows, stages }
  from "../run-report.js";

const SHELL = `
  <div class="panel">
    <p class="job-title" id="dTitle"></p>
    <p class="msg" id="dPath"></p>
    <div id="dNotes"></div>
    <video id="dVideo" controls preload="none"></video>
    <p class="hint hidden" id="dGone"></p>
    <div style="display:flex;gap:8px;margin:16px 0 4px;flex-wrap:wrap" id="dActions"></div>
    <details style="margin-top:14px" class="hidden" id="dSettingsBox">
      <summary>Settings this run used</summary>
      <div id="dSettings" style="margin-top:10px"></div>
    </details>
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
    // A running job reports this false because it has no output yet, but only a
    // finished one is ever drawn here.
    const here = job.output_exists !== false && !knownGone(job.output);
    this.$("#dTitle").textContent = (sample ? "Sample — " : "") + job.title;
    this.$("#dPath").textContent = job.message;

    const key = `${job.id}|${here}`;
    const vid = this.$("#dVideo"), gone = this.$("#dGone");
    if(vid.dataset.key !== key){
      vid.dataset.key = key;
      gone.textContent = sample ? SAMPLE_GONE : OUTPUT_GONE;
      vid.classList.toggle("hidden", !here);
      gone.classList.toggle("hidden", here);
      if(here){
        // The file can also go while the panel is open, which no flag fetched
        // earlier can know about.
        vid.onerror = () => {
          vid.classList.add("hidden");
          gone.classList.remove("hidden");
        };
        vid.src = `/api/job/${job.id}/video`;
      } else {
        vid.onerror = null;
        vid.removeAttribute("src");
      }
    }

    const actions = this.$("#dActions");
    if(actions.dataset.key !== key){
      actions.dataset.key = key;
      // Dub it promotes the sample without downloading again only while its
      // working files are still there — once they're gone the message already
      // says to run it again, so no button should compete with that.
      actions.innerHTML = (sample
        ? (here ? `<button class="primary" id="dEscalate">Dub it</button>` : "")
        : here ? `<button class="primary" id="dReveal">Show in Finder</button>` : "")
        + `<button class="ghost" id="dReset">Dub another</button>`;
      // Two clicks are one job — the id a full run gets is the same either
      // time, so this reuses the start-job path rather than a separate one.
      const escalate = this.$("#dEscalate");
      if(escalate) escalate.onclick = () => this.emit("start-job", {url: job.url, preview: false});
      const reveal = this.$("#dReveal");
      if(reveal) reveal.onclick = () => this.reveal(job);
      this.$("#dReset").onclick = () => this.emit("reset");
    }

    // A frame arrives about twice a second, and rebuilding the quality report
    // drops any selection made inside it — the one part of this panel meant to
    // be read and copied out of. Nothing below it changes again once a run has
    // finished.
    this.renderIfChanged(
      [job.id, job.status, job.finished, here, (s.voices || []).length,
       Object.keys(s.presets || {}).length, Object.keys(s.glossaries || {}).length],
      () => this.paint(job, s)
    );
  }

  paint(job, s){
    const sample = !!job.preview;
    const stats = job.stats || {};
    const yn = v => v === true ? "yes" : v === false ? "no" : "—";
    // pipeline.py tags a note "info" at the source when it explains something
    // or reports good news rather than a fault. Anything else — including
    // every note recorded before this tagging existed, a bare string — is a
    // warning, exactly as it always has been.
    const isInfo = n => n && typeof n === "object" && n.kind === "info";
    const text = n => isInfo(n) ? n.text : n;
    const notes = stats.notes || [];
    const warnings = notes.filter(n => !isInfo(n));
    this.$("#dNotes").innerHTML =
      notes.filter(isInfo).map(n=>`<p class="hint" style="margin:0 0 12px">${escapeHtml(text(n))}</p>`).join("")
      + (warnings.length
        ? `<div class="banner info" style="margin:0 0 12px">`
          + warnings.map(n=>`<div>${escapeHtml(text(n))}</div>`).join("") + `</div>`
        : "");

    const settings = runRows(job, s);
    this.$("#dSettingsBox").classList.toggle("hidden", !settings.length);
    this.$("#dSettings").innerHTML = statRows(settings);

    // What the dub came out like, not what the machinery did on the way. The
    // engineering numbers this used to carry — frame counts, sub-second A/V
    // drift, peak dB, working files freed — each already have somewhere better
    // to be: a re-encoded picture and a silent track both raise a note above,
    // and the rest is in the diagnostics report. Sixteen rows for a finished
    // video answered a question nobody had.
    this.$("#dStats").innerHTML = statRows([
      ...(sample ? [["Sample taken from", stats.preview_from ?? "0:00"]] : []),
      ["Speakers found", stats.speakers ?? "—"],
      ["Voices used", stats.voices ?? "—"],
      ["Translated by", stats.translated_by ?? "—"],
      ["Music and effects kept", yn(stats.music_kept)],
      ["Lines spoken", stats.lines_spoken ?? "—"],
      ["Lines compressed to fit", `${stats.compressed ?? 0} of ${stats.lines ?? 0}`],
      ["Hardest squeeze", stats.max_factor ? stats.max_factor.toFixed(2)+"x" : "none"],
      ["Video with no dubbed line", stats.no_line_seconds == null ? "—"
        : `${Math.round((stats.no_line_share ?? 0)*100)}% `
          + `(${stats.no_line_seconds}s)`],
      ["Took", fmt(job.elapsed)]
    ]) + stages(job);
  }

  // Asked again at the click, because the flag saying the file is there was
  // answered when the run finished and /api/reveal on a path that has gone
  // opens nothing and says nothing.
  async reveal(job){
    if(await outputGone(job.output)){
      this.update(store.state);
      return;
    }
    this.emit("reveal", {path: job.output});
  }
}

customElements.define("done-panel", DonePanel);
