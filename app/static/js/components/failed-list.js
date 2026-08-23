import { CappedList } from "./capped-list.js";
import { store } from "../store.js";
import { escapeHtml, link, dayAndClockText } from "../format.js";
import { presetLabel } from "../run-report.js";

const SHOWN = 6;

const SHELL = `
  <style>
    .head{display:flex;align-items:flex-start;justify-content:space-between;
          gap:14px;flex-wrap:wrap;margin-bottom:16px}
    .head > div{flex:1;min-width:200px}
    .head p{margin:5px 0 0}
  </style>
  <div id="wrap"></div>
`;

// Behind the row's own disclosure rather than the subtitle: the reason is
// short enough to read at a glance, but a translator or a tool can hand back
// several paragraphs, and a list of runs is not the place for those to sit
// open by default.
function detail(job, s){
  const said = (job.error_detail || "").trim();
  return `
    ${job.url ? `<p class="hint" style="margin:0 0 12px;word-break:break-word">From `
                + `${link(job.url)}</p>` : ""}
    ${job.preset ? `<p class="hint" style="margin:0 0 12px">Ran with the `
                  + `${escapeHtml(presetLabel(job.preset, s.presets))} preset. Try `
                  + `again uses the current settings.</p>` : ""}
    ${said
      ? `<p style="margin:0 0 6px;font-weight:600">Error details</p>`
        + `<pre style="white-space:pre-wrap;overflow-wrap:anywhere;margin:0">`
        + `${escapeHtml(said)}</pre>`
      : `<p class="hint" style="margin:0">No further details were recorded.</p>`}
  `;
}

class FailedList extends CappedList {
  connectedCallback(){
    super.connectedCallback();
    this.html(SHELL);
    this._unsub = store.subscribe(s => this.update(s));
  }

  update(s){
    // Whether this whole component is on screen belongs to manage-panel's tab
    // switch, not to any count kept here — a failure must never pull the tab
    // out from under whatever the user is looking at, so this only ever
    // decides what fills the space the tab has already given it.
    let count = 0, newest = 0;
    for(const j of Object.values(s.jobs)){
      if(j.status !== "error") continue;
      count++;
      const at = j.finished || j.started || 0;
      if(at > newest) newest = at;
    }
    this.renderIfChanged([count, newest], () => this.paint(s));
  }

  paint(s){
    const failed = Object.values(s.jobs)
      .filter(j => j.status === "error")
      .sort((a,b) => (b.finished || b.started || 0) - (a.finished || a.started || 0));

    const wrap = this.$("#wrap");
    // Collapses to nothing rather than an empty state: unlike the dubbed-videos
    // list, which is always the answer to "where will my video go", a healthy
    // app that has never failed a run has nothing here worth a permanent row of
    // its own taking up room above that list.
    if(!failed.length){
      wrap.innerHTML = "";
      this._rows = new Map();
      this._openId = null;
      return;
    }
    if(!this.$("#failed")){
      wrap.innerHTML = `
        <div class="panel quiet">
          <div class="head">
            <div>
              <h2 class="job-title">Failed runs</h2>
            </div>
          </div>
          <div id="failed"></div>
        </div>
      `;
    }

    this.paintRows(failed, SHOWN, this.$("#failed"), "moreFailed",
      rest => `…and ${rest} earlier failure${rest === 1 ? "" : "s"}.`,
      j => {
        const reason = j.error || "The run failed.";
        const data = {
          color: "var(--bad)",
          title: j.title || j.url,
          badge: j.preview ? "sample" : "",
          subtitle: `${dayAndClockText(j.finished || j.started)} — ${reason}`,
          detail: detail(j, s),
          actions: [
            {label: "Try again", className: "primary",
             onClick: () => this.emit("start-job", {url: j.url, preview: j.preview})},
            // Same call as the one in Settings and on the live failed card: the
            // failure is already in the recent log entries, tagged with its id,
            // so no job argument is needed here either.
            {label: "Diagnostics",
             onClick: () => document.querySelector("diagnostics-panel")?.open()},
          ],
        };
        return {
          id: j.id,
          sig: JSON.stringify([j.id, j.status, j.error, j.error_detail,
                               data.title, data.subtitle]),
          data,
        };
      });
  }
}

customElements.define("failed-list", FailedList);
