import { BaseElement } from "../base-element.js";
import { store } from "../store.js";
import { fmt, friendlyFolder, escapeHtml, escapeAttr, niceDate } from "../format.js";
import { OUTPUT_GONE, knownGone, outputGone, runRows, statRows, stages }
  from "../run-report.js";
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
  <div class="panel quiet">
    <div class="head">
      <div>
        <h2>Your dubbed videos</h2>
        <p class="hint" id="outputWhere">Saved to your Movies folder.</p>
      </div>
      <button class="ghost" id="openFolder">Open folder</button>
    </div>
    <div id="history"></div>
  </div>
`;

// The stamp is epoch seconds and niceDate takes a date string, so the day is
// assembled from local parts: a UTC one is the day after for anything finished
// in the evening. A run from another year says which one.
function finishedOn(job){
  const when = new Date((job.finished || 0) * 1000);
  if(!job.finished || isNaN(when)) return null;
  const pad = n => String(n).padStart(2, "0");
  const day = niceDate(`${when.getFullYear()}-${pad(when.getMonth()+1)}-${pad(when.getDate())}`);
  const year = when.getFullYear() === new Date().getFullYear() ? "" : ` ${when.getFullYear()}`;
  return {day: `${day}${year}`,
          clock: when.toLocaleTimeString(undefined, {hour: "numeric", minute: "2-digit"})};
}

function link(url){
  const shown = escapeHtml(String(url).replace(/^https?:\/\/(www\.)?/i, ""));
  return /^https?:\/\//i.test(url)
    ? `<a href="${escapeAttr(url)}" target="_blank" rel="noopener"
         style="color:var(--accent)">${shown}</a>`
    : shown;
}

function detail(job, state, gone){
  const folder = job.output ? job.output.split("/").slice(0, -1).join("/") : "";
  const when = finishedOn(job);
  const used = runRows(job, state, {outcomes: true});
  return `
    <p style="margin:0 0 7px;font-weight:600">${escapeHtml(job.title || "")}</p>
    ${gone ? `<p class="hint" style="margin:0 0 10px">${OUTPUT_GONE}</p>` : ""}
    ${job.url ? `<p class="hint" style="margin:0 0 12px;word-break:break-word">From `
                + `${link(job.url)}</p>` : ""}
    ${statRows([["Took", fmt(job.elapsed)]])}
    ${stages(job)}
    ${statRows([
      ...(when ? [["Finished", `${when.day} at ${when.clock}`]] : []),
      ...(folder ? [[gone ? "Was saved in" : "Saved in", friendlyFolder(folder)]] : []),
    ])}
    ${used.length ? `<p class="hint" style="margin:16px 0 2px;font-weight:600">What this `
                    + `run used</p>${statRows(used)}` : ""}
  `;
}

class HistoryList extends BaseElement {
  connectedCallback(){
    this.html(SHELL);
    // Always here, and answering the question before it is asked: when empty it
    // says where finished videos will go, and when not it is where they are.
    this.$("#openFolder").onclick = () =>
      this.emit("reveal", {path: store.state.output_dir || ""});
    this._rows = new Map();
    this._unsub = store.subscribe(s => this.update(s));
  }

  disconnectedCallback(){
    this._unsub?.();
  }

  update(s){
    // Samples are excluded for the same reason they never reach the videos
    // folder: a sample's file is a working intermediate meant to be thrown away.
    // A finished run stays "current" — the success card above keeps pointing
    // at it — for as long as nothing has taken its place there, which can be
    // indefinitely if nobody starts another job. This list is where "it's
    // done" gets confirmed, so it has to carry the run from the instant it
    // lands, not from whenever the user happens to move on from that card.
    // Counted rather than collected, because a frame arrives every second while
    // a job runs and this has to be cheaper than the paint it is turning away:
    // a run that has finished has nothing left to change but its file.
    let listed = 0, gone = 0, newest = 0;
    for(const j of Object.values(s.jobs)){
      if(j.status !== "done" || j.preview) continue;
      listed++;
      if(j.output_exists === false || knownGone(j.output)) gone++;
      if(j.started > newest) newest = j.started;
    }
    this.renderIfChanged(
      [listed, gone, newest, s.output_dir, (s.voices || []).length,
       Object.keys(s.presets || {}).length, Object.keys(s.glossaries || {}).length],
      () => this.paint(s)
    );
  }

  paint(s){
    const past = Object.values(s.jobs)
      .filter(j => j.status === "done" && !j.preview)
      .sort((a,b) => b.started - a.started);

    this.$("#outputWhere").textContent = past.length
      ? `Saved to ${friendlyFolder(s.output_dir)}.`
      : `Nothing dubbed yet. Finished videos are saved to `
        + `${friendlyFolder(s.output_dir)}.`;

    const box = this.$("#history");
    const kept = new Map();
    // Capped, because this is a shortcut to the recent ones, not a file
    // manager — Open folder is the answer for the rest.
    past.slice(0, SHOWN).forEach(j => {
      const gone = j.output_exists === false || knownGone(j.output);
      const data = {
        title: j.title,
        subtitle: [(finishedOn(j) || {}).day, fmt(j.elapsed),
                   ...(gone ? ["file no longer there"] : [])].filter(Boolean).join(" · "),
        detail: detail(j, s, gone),
        // Nothing to show: the file has been moved or deleted, and offering to
        // reveal it would open the folder on nothing.
        actions: gone ? [] : [{label: "Show", onClick: () => this.reveal(j)}],
      };
      const sig = JSON.stringify([j.id, j.output, data.title, data.subtitle, data.detail]);
      const had = this._rows.get(j.id);
      const row = had ? had.row : document.createElement("list-row");
      // A row is left alone when what it would say is what it already says.
      // Handing it the same content again would take the focus off its
      // disclosure button and drop any selection made in an open one.
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

    const rest = past.length - SHOWN;
    let note = this.$("#more");
    if(rest > 0 && !note){
      note = document.createElement("p");
      note.id = "more";
      note.className = "hint";
      box.appendChild(note);
    }
    if(rest > 0) note.textContent = `…and ${rest} more in the folder.`;
    else if(note) note.remove();
  }

  // Asked again at the click, because whether the file is still there was
  // answered when the list was fetched and /api/reveal on a path that has gone
  // opens nothing and says nothing.
  async reveal(j){
    if(await outputGone(j.output)){
      this.update(store.state);
      return;
    }
    this.emit("reveal", {path: j.output});
  }

  // One at a time: several open at once turns a list of six into a page of
  // detail with no list left in it.
  opened(id){
    this._openId = id;
    this._rows.forEach(({row}, key) => { if(key !== id) row.setOpen(false); });
  }
}

customElements.define("history-list", HistoryList);
