import { BaseElement } from "../base-element.js";
import { store } from "../store.js";
import { escapeHtml, escapeAttr, niceDate } from "../format.js";
import "./list-row.js";

const SHOWN = 6;

const SHELL = `
  <style>
    .head{display:flex;align-items:flex-start;justify-content:space-between;
          gap:14px;flex-wrap:wrap;margin-bottom:16px}
    .head > div{flex:1;min-width:200px}
    .head h2{margin:0;font-size:17px;font-weight:650;letter-spacing:-.01em}
    .head p{margin:5px 0 0}
  </style>
  <div id="wrap"></div>
`;

// The stamp is epoch seconds and niceDate takes a date string, so the day is
// assembled from local parts the same way the dubbed-videos list already does
// it — a UTC reading would put an evening failure on the day after.
function when(job){
  const at = new Date((job.finished || job.started || 0) * 1000);
  if(!(job.finished || job.started) || isNaN(at)) return "";
  const pad = n => String(n).padStart(2, "0");
  const day = niceDate(`${at.getFullYear()}-${pad(at.getMonth()+1)}-${pad(at.getDate())}`);
  const year = at.getFullYear() === new Date().getFullYear() ? "" : ` ${at.getFullYear()}`;
  const clock = at.toLocaleTimeString(undefined, {hour: "numeric", minute: "2-digit"});
  return `${day}${year} at ${clock}`;
}

function link(url){
  const shown = escapeHtml(String(url).replace(/^https?:\/\/(www\.)?/i, ""));
  return /^https?:\/\//i.test(url)
    ? `<a href="${escapeAttr(url)}" target="_blank" rel="noopener"
         style="color:var(--accent)">${shown}</a>`
    : shown;
}

// A preset name, in the same words the quality picker uses — falling back to
// a capitalised raw value for a preset the shipped list has since dropped,
// the same fallback run-report.js uses for a finished run's settings.
function presetLabel(preset, presets){
  const found = (presets || {})[preset];
  if(found) return found.label;
  return String(preset).charAt(0).toUpperCase() + String(preset).slice(1);
}

// Behind the row's own disclosure rather than the subtitle: the reason is
// short enough to read at a glance, but a translator or a tool can hand back
// several paragraphs, and a list of runs is not the place for those to sit
// open by default.
function detail(job, s){
  const said = (job.error_detail || "").trim();
  return `
    ${job.url ? `<p class="hint" style="margin:0 0 12px;word-break:break-word">From `
                + `${link(job.url)}</p>` : ""}
    ${job.preset ? `<p class="hint" style="margin:0 0 12px">This attempt used the `
                  + `${escapeHtml(presetLabel(job.preset, s.presets))} preset. Try `
                  + `again uses whatever Settings holds now, which may differ.</p>` : ""}
    ${said
      ? `<p style="margin:0 0 6px;font-weight:600">What the error actually said</p>`
        + `<pre style="white-space:pre-wrap;overflow-wrap:anywhere;margin:0">`
        + `${escapeHtml(said)}</pre>`
      : `<p class="hint" style="margin:0">Nothing more specific was recorded for `
        + `this attempt.</p>`}
  `;
}

class FailedList extends BaseElement {
  connectedCallback(){
    this.html(SHELL);
    this._rows = new Map();
    this._unsub = store.subscribe(s => this.update(s));
  }

  disconnectedCallback(){
    this._unsub?.();
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
              <h2>Runs that didn't finish</h2>
              <p class="hint">Kept here so a failure survives closing the app, not
                only reloading the page.</p>
            </div>
          </div>
          <div id="failed"></div>
        </div>
      `;
    }

    const box = this.$("#failed");
    const kept = new Map();
    failed.slice(0, SHOWN).forEach(j => {
      const reason = j.error || "Something went wrong.";
      const data = {
        color: "var(--bad)",
        title: j.title || j.url,
        badge: j.preview ? "sample" : "",
        subtitle: `${when(j)} — ${reason}`,
        detail: detail(j, s),
        actions: [
          {label: "Try again", className: "primary",
           onClick: () => this.emit("start-job", {url: j.url, preview: j.preview})},
          // Same call as the one in Settings and on the live failed card: the
          // failure is already in the recent log entries, tagged with its id,
          // so no job argument is needed here either.
          {label: "Copy details",
           onClick: () => document.querySelector("diagnostics-panel")?.open()},
        ],
      };
      const sig = JSON.stringify([j.id, j.status, j.error, j.error_detail,
                                  data.title, data.subtitle]);
      const had = this._rows.get(j.id);
      const row = had ? had.row : document.createElement("list-row");
      // Left alone when nothing it would say has changed, the same as the
      // dubbed-videos list: rewriting it drops the focus off its disclosure
      // button and closes it if it was open.
      if(!had || had.sig !== sig){
        row.data = {...data, open: j.id === this._openId,
                    onToggle: open => this.opened(open ? j.id : null)};
      }
      kept.set(j.id, {row, sig});
    });
    // Gone first, then placed: a row put in position ahead of a row that is
    // about to be removed would be moved twice, and moving one is what takes
    // the focus out of it.
    this._rows.forEach(({row}, id) => { if(!kept.has(id)) row.remove(); });
    this._rows = kept;
    [...kept.values()].forEach(({row}, at) => {
      const there = box.children[at];
      if(there !== row) box.insertBefore(row, there || null);
    });
    if(!this._rows.has(this._openId)) this._openId = null;

    const rest = failed.length - SHOWN;
    let note = this.$("#moreFailed");
    if(rest > 0 && !note){
      note = document.createElement("p");
      note.id = "moreFailed";
      note.className = "hint";
      box.appendChild(note);
    }
    if(rest > 0){
      note.textContent = `…and ${rest} earlier failure${rest === 1 ? "" : "s"}, not shown here.`;
    } else if(note){
      note.remove();
    }
  }

  // One at a time: several open at once turns a list of six into a page of
  // detail with no list left in it.
  opened(id){
    this._openId = id;
    this._rows.forEach(({row}, key) => { if(key !== id) row.setOpen(false); });
  }
}

customElements.define("failed-list", FailedList);
