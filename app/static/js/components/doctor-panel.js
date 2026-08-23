import { BaseElement } from "../base-element.js";
import { store } from "../store.js";
import { escapeHtml } from "../format.js";

const SHELL = `
  <div class="panel quiet">
    <h2 class="job-title">Setup check</h2>
    <p id="verdict" style="margin:2px 0 14px;font-weight:600"></p>
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
      const verdict = this.$("#verdict");
      verdict.classList.toggle("hidden", !s.doctor);
      if(!s.doctor) return;
      // Nine rows of technical names never answered the one question this
      // panel exists to answer. `ready` was already computed for the tab
      // picker; this is the same value, said in words.
      verdict.textContent = s.doctor.ready
        ? "Ready. Everything needed is installed."
        : "Not ready. A check below needs fixing — it says what to do.";
      verdict.style.color = s.doctor.ready ? "var(--ok)" : "var(--bad)";
      this.$("#doctor").innerHTML = s.doctor.checks.map(c=>`
        <div class="check">
          <span class="dot ${c.ok?"ok":(c.optional?"opt":"bad")}"></span>
          <div><b>${escapeHtml(c.name)}</b>${c.optional&&!c.ok?'<span class="tag">optional</span>':""}
            ${c.ok?"":`<br><code>${escapeHtml(c.hint)}</code>`}
            ${c.note?`<br><code>${escapeHtml(c.note)}</code>`:""}</div>
        </div>`).join("");
    });
  }

  paintMachine(m){
    const bits = [];
    if(m?.engine) bits.push(m.engine);
    if(m?.ram_gb) bits.push(`${m.ram_gb} GB memory`);
    this.$("#machine").textContent = bits.join(" · ");
    this.$("#machine").classList.toggle("hidden", !bits.length);
  }
}

customElements.define("doctor-panel", DoctorPanel);
