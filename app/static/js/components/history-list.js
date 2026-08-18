import { BaseElement } from "../base-element.js";
import { store } from "../store.js";
import { fmt, friendlyFolder } from "../format.js";
import "./list-row.js";

const SHOWN = 6;

const SHELL = `
  <div class="panel">
    <div class="danger-row" style="border-top:none;margin:0;padding:0">
      <div style="min-width:0">
        <b style="font-size:15px">Your dubbed videos</b>
        <small id="outputWhere">Saved to your Movies folder.</small>
      </div>
      <button class="primary" id="openFolder">Open folder</button>
    </div>
    <div id="history" style="margin-top:8px"></div>
  </div>
`;

class HistoryList extends BaseElement {
  connectedCallback(){
    this.html(SHELL);
    // Always here, and answering the question before it is asked: when empty it
    // says where finished videos will go, and when not it is where they are.
    this.$("#openFolder").onclick = () =>
      this.emit("reveal", {path: store.state.output_dir || ""});
    this._unsub = store.subscribe(s => this.update(s));
  }

  disconnectedCallback(){
    this._unsub?.();
  }

  update(s){
    // Samples are excluded for the same reason they never reach the videos
    // folder: a sample's file is a working intermediate meant to be thrown away.
    const past = Object.values(s.jobs)
      .filter(j => j.status === "done" && j.id !== s.current && !j.preview)
      .sort((a,b) => b.started - a.started);

    this.renderIfChanged(
      past.map(j => j.id).join("|"),
      () => {
        this.$("#outputWhere").textContent = past.length
          ? `Saved to ${friendlyFolder(s.output_dir)}.`
          : `Nothing dubbed yet. Finished videos are saved to `
            + `${friendlyFolder(s.output_dir)}.`;

        const box = this.$("#history");
        box.innerHTML = "";
        // Capped, because this is a shortcut to the recent ones, not a file
        // manager — Open folder is the answer for the rest.
        past.slice(0, SHOWN).forEach(j => {
          const row = document.createElement("list-row");
          row.data = {
            title: j.title,
            subtitle: fmt(j.elapsed),
            actions: [{label: "Show", onClick: () => this.emit("reveal", {path: j.output})}],
          };
          box.appendChild(row);
        });
        if(past.length > SHOWN){
          const note = document.createElement("p");
          note.className = "hint";
          note.textContent = `…and ${past.length - SHOWN} more in the folder.`;
          box.appendChild(note);
        }
      }
    );
  }
}

customElements.define("history-list", HistoryList);
