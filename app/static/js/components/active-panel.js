import { BaseElement } from "../base-element.js";
import { store } from "../store.js";
import { escapeHtml, fmt, fmtRough } from "../format.js";
import { STAGE_LABELS, DEFAULT_STAGES } from "../stages.js";

const SHELL = `
  <div class="panel">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:14px">
      <div style="min-width:0">
        <p class="job-title" id="aTitle">Starting…</p>
        <p class="msg" id="aMsg" role="status" aria-live="polite">Getting ready</p>
      </div>
      <button class="ghost" id="cancelBtn">Cancel</button>
    </div>
    <div class="bar"><i id="aBar"></i></div>
    <div style="display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;
                font-size:13px;color:var(--muted)">
      <span id="aPct">0%</span><span id="aElapsed"></span>
    </div>
    <div class="stages" id="aStages"></div>
    <div id="aRefs"></div>
  </div>
`;

class ActivePanel extends BaseElement {
  connectedCallback(){
    this.html(SHELL);
    this.$("#cancelBtn").onclick = () => {
      const job = store.state.jobs[store.state.current];
      if(job) this.emit("cancel-job", {id: job.id});
    };
    this._unsub = store.subscribe(s => this.update(s));
  }

  disconnectedCallback(){
    this._unsub?.();
  }

  update(s){
    const job = s.jobs[s.current];
    const running = !!job && (job.status === "running" || job.status === "queued");
    this.hidden = !running;
    if(!running) return;

    this.$("#aTitle").textContent = (job.preview ? "Sample — " : "")
                            + (job.title || "Working…");
    this.say("#aMsg", job.message);
    this.$("#aBar").style.width = (job.overall*100).toFixed(1)+"%";
    this.$("#aPct").textContent = Math.round(job.overall*100)+"%";

    // How long is left is the question this app raises above every other, and
    // the answer is withheld until there is enough behind us to extrapolate
    // from: loading the models alone can be a third of a short job, so anything
    // earlier swings wildly and reads as the app not knowing what it is doing.
    const parts = [];
    if(job.elapsed) parts.push(fmt(job.elapsed)+" elapsed");
    if(job.overall > 0.12 && job.elapsed > 20){
      parts.push("about "+fmtRough(job.elapsed*(1-job.overall)/job.overall)+" left");
    }
    this.$("#aElapsed").textContent = parts.join(" · ");

    const stages = (job.stages && job.stages.length) ? job.stages : DEFAULT_STAGES;
    const at = stages.indexOf(job.stage);
    const times = job.stage_times || {};
    this.$("#aStages").innerHTML = stages.map((key,i)=>{
      const spent = i===at ? (times[key]||0) + (job.stage_elapsed||0)
                  : (i<at ? times[key] : 0);
      const clock = spent > 0 ? `<small>${fmt(spent)}</small>` : "";
      return `<span class="chip ${i===at?"on":(i<at?"done":"")}">${
        escapeHtml(STAGE_LABELS[key] || key)}${clock}</span>`;
    }).join("");

    // A cancel can only take effect between steps, so say it has been heard
    // rather than leaving the button looking unpressed for several minutes.
    this.$("#cancelBtn").disabled = !!job.cancelling;
    this.$("#cancelBtn").textContent = job.cancelling ? "Stopping…" : "Cancel";

    this.renderReferences(job);
  }

  renderReferences(job){
    const box = this.$("#aRefs");
    const refs = job.references || [];
    // Only redraw when the set actually changes: rewriting innerHTML destroys
    // and recreates the <audio> elements, which restarts any clip being played.
    const key = refs.join("|");
    if(box.dataset.refs === key) return;
    box.dataset.refs = key;
    box.innerHTML = refs.length
      ? `<p class="hint" style="margin:10px 0 4px">Cloning from ${refs.length} `
        + `reference clip${refs.length===1?"":"s"} taken from the original. Listen `
        + `now — a bad one colours every line.</p>`
        + refs.map((_,i)=>`<audio controls preload="none" style="width:100%;margin-top:6px"
             src="/api/job/${encodeURIComponent(job.id)}/reference/${i}"></audio>`).join("")
      : "";
  }
}

customElements.define("active-panel", ActivePanel);
