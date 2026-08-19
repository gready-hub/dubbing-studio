import { BaseElement } from "../base-element.js";
import { store } from "../store.js";
import { escapeHtml } from "../format.js";

const SHELL = `
  <div class="panel">
    <b class="job-title">Setup check</b>
    <p class="hint hidden" id="machine" style="margin-bottom:0"></p>
    <div id="doctor" style="margin-top:10px"></div>
  </div>
`;

class DoctorPanel extends BaseElement {
  connectedCallback(){
    this.html(SHELL);
    this._unsub = store.subscribe(s => this.update(s));
  }

  disconnectedCallback(){
    this._unsub?.();
  }

  update(s){
    this.renderIfChanged([s.machine, s.doctor], () => {
      this.paintMachine(s.machine);
      if(!s.doctor) return;
      this.$("#doctor").innerHTML = s.doctor.checks.map(c=>`
        <div class="check">
          <span class="dot ${c.ok?"ok":(c.optional?"opt":"bad")}"></span>
          <div><b>${escapeHtml(c.name)}</b>${c.optional&&!c.ok?'<span class="tag">optional</span>':""}
            ${c.ok?"":`<br><code>${escapeHtml(c.hint)}</code>`}</div>
        </div>`).join("");
    });
  }

  paintMachine(m){
    const bits = [];
    if(m?.engine) bits.push(m.engine);
    if(m?.ram_gb) bits.push(`${m.ram_gb} GB memory`);
    if(m?.in_docker) bits.push("in Docker");
    this.$("#machine").textContent = bits.join(" · ");
    this.$("#machine").classList.toggle("hidden", !bits.length);
  }
}

customElements.define("doctor-panel", DoctorPanel);
