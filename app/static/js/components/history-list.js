import { CappedList } from "./capped-list.js";
import { store } from "../store.js";
import { fmt, friendlyFolder, escapeHtml, dayAndClock, link } from "../format.js";
import { OUTPUT_GONE, knownGone, outputGone, runRows, statRows, stages }
  from "../run-report.js";

const SHOWN = 6;

const SHELL = `
  <style>
    .head{display:flex;align-items:flex-start;justify-content:space-between;
          gap:14px;flex-wrap:wrap;margin-bottom:16px}
    .head > div{flex:1;min-width:200px}
    .head p{margin:5px 0 0}
  </style>
  <div class="panel quiet">
    <div class="head">
      <div>
        <h2 class="job-title">Your dubbed videos</h2>
        <p class="hint" id="outputWhere">Saved to your Movies folder.</p>
      </div>
      <button class="ghost" id="openFolder">Open folder</button>
    </div>
    <div id="history"></div>
  </div>
`;

function detail(job, state, gone){
  const folder = job.output ? job.output.split("/").slice(0, -1).join("/") : "";
  const when = dayAndClock(job.finished);
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

class HistoryList extends CappedList {
  connectedCallback(){
    super.connectedCallback();
    this.html(SHELL);
    // Always here, and answering the question before it is asked: when empty it
    // says where finished videos will go, and when not it is where they are.
    this.$("#openFolder").onclick = () =>
      this.emit("reveal", {path: store.state.output_dir || ""});
    this._unsub = store.subscribe(s => this.update(s));
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

    // Capped, because this is a shortcut to the recent ones, not a file
    // manager — Open folder is the answer for the rest.
    this.paintRows(past, SHOWN, this.$("#history"), "more",
      rest => `…and ${rest} more in the folder.`,
      j => {
        const gone = j.output_exists === false || knownGone(j.output);
        const data = {
          title: j.title,
          // Bare beside a title, a duration reads as the video's own length —
          // it is how long the dub took. "Took", the same word the detail rows
          // below and the sample report use for the same number, says so.
          subtitle: [(dayAndClock(j.finished) || {}).day, `Took ${fmt(j.elapsed)}`,
                     ...(gone ? ["file no longer there"] : [])].filter(Boolean).join(" · "),
          detail: detail(j, s, gone),
          // Nothing to show: the file has been moved or deleted, and offering to
          // reveal it would open the folder on nothing.
          actions: gone ? [] : [{label: "Show", onClick: () => this.reveal(j)}],
        };
        return {
          id: j.id,
          sig: JSON.stringify([j.id, j.output, data.title, data.subtitle, data.detail]),
          data,
        };
      });
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
}

customElements.define("history-list", HistoryList);
