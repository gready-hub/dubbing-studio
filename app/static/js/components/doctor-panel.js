import { BaseElement } from "../base-element.js";
import { store } from "../store.js";
import { escapeHtml } from "../format.js";

const SHELL = `
  <div class="panel">
    <b style="font-size:15px">Setup check</b>
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
    if(!s.doctor) return;
    const d = s.doctor;
    this.$("#doctor").innerHTML = d.checks.map(c=>`
      <div class="check">
        <span class="dot ${c.ok?"ok":(c.optional?"opt":"bad")}"></span>
        <div><b>${escapeHtml(c.name)}</b>${c.optional&&!c.ok?'<span class="tag">optional</span>':""}
          ${c.ok?"":`<br><code>${escapeHtml(c.hint)}</code>`}</div>
      </div>`).join("");
  }
}

customElements.define("doctor-panel", DoctorPanel);
