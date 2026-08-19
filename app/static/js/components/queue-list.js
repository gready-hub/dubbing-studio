import { BaseElement } from "../base-element.js";
import { store } from "../store.js";
import "./list-row.js";

const SHELL = `
  <div class="panel">
    <h2 class="job-title">Waiting to start</h2>
    <p class="hint">One video is dubbed at a time — two at once only makes both
      slower.</p>
    <div id="queue" style="margin-top:6px"></div>
  </div>
`;

class QueueList extends BaseElement {
  connectedCallback(){
    this.html(SHELL);
    this._unsub = store.subscribe(s => this.update(s));
  }

  disconnectedCallback(){
    this._unsub?.();
  }

  update(s){
    const waiting = Object.values(s.jobs)
      .filter(j => j.status === "queued")
      .sort((a,b) => (a.queue_position||0) - (b.queue_position||0));

    this.renderIfChanged(
      waiting.map(j => `${j.id}:${j.message}`).join("|"),
      () => {
        this.hidden = waiting.length === 0;
        const box = this.$("#queue");
        box.innerHTML = "";
        waiting.forEach(j => {
          const row = document.createElement("list-row");
          // Tagged, because a sample and the full run of the same link are two
          // rows with the same title, and without it the waiting list reads as
          // a duplicate.
          row.data = {
            title: j.title || j.url,
            subtitle: j.message,
            badge: j.preview ? "sample" : "",
            actions: [{label: "Stop", onClick: () => this.emit("cancel-job", {id: j.id})}],
          };
          box.appendChild(row);
        });
      }
    );
  }
}

customElements.define("queue-list", QueueList);
