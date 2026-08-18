import { BaseElement } from "../base-element.js";
import { store } from "../store.js";

const STEPS = ["Setup", "Running", "Done"];

class StepIndicator extends BaseElement {
  connectedCallback(){
    this.html(`<div class="stages" id="steps"></div>`);
    this._unsub = store.subscribe(s => this.update(s));
  }

  disconnectedCallback(){
    this._unsub?.();
  }

  update(s){
    const job = s.jobs[s.current];
    this.hidden = !job;
    if(!job) return;
    const at = job.status === "done" ? 2 : 1;
    this.$("#steps").innerHTML = STEPS.map((label, i) =>
      `<span class="chip ${i===at?"on":(i<at?"done":"")}">${label}</span>`).join("");
  }
}

customElements.define("step-indicator", StepIndicator);
