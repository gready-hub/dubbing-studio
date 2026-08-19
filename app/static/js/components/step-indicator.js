import { BaseElement } from "../base-element.js";
import { store } from "../store.js";

const STEPS = ["Setup", "Running", "Done"];

const HALTED = {
  error:     {cls: "failed",  label: "Failed"},
  cancelled: {cls: "stopped", label: "Stopped"},
};

const SHELL = `<div class="stages" id="steps"></div>`;

class StepIndicator extends BaseElement {
  connectedCallback(){
    this.html(SHELL);
    this._unsub = store.subscribe(s => this.update(s));
  }

  disconnectedCallback(){
    this._unsub?.();
  }

  update(s){
    const job = s.jobs[s.current];
    this.hidden = !job;
    if(!job) return;
    const halted = HALTED[job.status];
    const at = job.status === "done" ? 2 : 1;
    this.$("#steps").innerHTML = STEPS.map((label, i) => {
      if(i !== at) return `<span class="chip ${i < at ? "done" : ""}">${label}</span>`;
      return `<span class="chip ${halted ? halted.cls : "on"}">${
        halted ? halted.label : label}</span>`;
    }).join("");
  }
}

customElements.define("step-indicator", StepIndicator);
