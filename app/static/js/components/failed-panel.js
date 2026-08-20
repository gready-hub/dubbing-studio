import { BaseElement } from "../base-element.js";
import { store } from "../store.js";

const SHELL = `
  <div class="panel">
    <p class="job-title" id="fTitle"></p>
    <div class="banner bad" id="fMsg" style="margin:10px 0 0" role="status" aria-live="polite"></div>
    <details id="fDetailWrap" class="hidden" style="margin-top:4px">
      <summary>What the error actually said</summary>
      <pre id="fDetail"></pre>
    </details>
    <div style="display:flex;gap:8px;margin-top:16px;flex-wrap:wrap">
      <button class="primary" id="fRetry">Try again</button>
      <button class="ghost" id="fReset">Dub another</button>
      <button class="ghost" id="fCopy">Copy details</button>
    </div>
  </div>
`;

class FailedPanel extends BaseElement {
  connectedCallback(){
    this.html(SHELL);
    // Same link, same kind of run: the previous attempt is in an error state,
    // so this builds a fresh job over it rather than handing back the dead one.
    this.$("#fRetry").onclick = () => {
      const job = store.state.jobs[store.state.current];
      if(job) this.emit("start-job", {url: job.url, preview: job.preview});
    };
    this.$("#fReset").onclick = () => this.emit("reset");
    // Same call as the one in the header: the failure is already in the recent
    // log entries, tagged with its id, so no job argument is needed.
    this.$("#fCopy").onclick = () => document.querySelector("diagnostics-panel")?.open();
    this._unsub = store.subscribe(s => this.update(s));
  }

  disconnectedCallback(){
    this._unsub?.();
  }

  update(s){
    const job = s.jobs[s.current];
    const isFailed = !!job && job.status === "error";
    this.hidden = !isFailed;
    if(!isFailed) return;

    this.$("#fTitle").textContent = job.title
      ? (job.preview ? "Sample — " : "") + job.title
      : "That video couldn't be dubbed";
    this.say("#fMsg", job.error || "Something went wrong.");
    // The tool's own words: one click away and copyable, rather than buried in
    // a log under Application Support.
    const detail = (job.error_detail || "").trim();
    this.$("#fDetailWrap").classList.toggle("hidden", !detail);
    this.$("#fDetail").textContent = detail;
  }
}

customElements.define("failed-panel", FailedPanel);
