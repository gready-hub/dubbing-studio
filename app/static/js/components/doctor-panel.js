import { BaseElement } from "../base-element.js";
import { store } from "../store.js";
import { escapeHtml } from "../format.js";
import { api } from "../api.js";

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
            ${c.note?`<br><code>${escapeHtml(c.note)}</code>`:""}
            ${c.action==="update-ytdlp"?`<br><button class="small" data-action="update-ytdlp"
                 style="margin-top:6px">Update yt-dlp</button>
               <span class="hint" data-said="update-ytdlp"></span>`:""}</div>
        </div>`).join("");
      this.wireYtdlpUpdate();
    });
  }

  // The one check the app can fix by itself. yt-dlp goes stale on YouTube's
  // schedule rather than ours, and until this button existed the only remedy
  // on offer was a full reinstall in a Terminal window — a price high enough
  // that the correct advice went untaken and the 403 came back instead.
  wireYtdlpUpdate(){
    const button = this.$('[data-action="update-ytdlp"]');
    if(!button) return;
    const said = this.$('[data-said="update-ytdlp"]');
    button.onclick = async () => {
      button.disabled = true;
      said.textContent = " Updating…";
      try {
        const r = await api.updateYtdlp();
        // Already current is a success, and saying so beats a silent button:
        // somebody pressing this is trying to fix a download that just failed,
        // and needs to know whether to look elsewhere.
        said.textContent = r.changed ? ` Updated to ${r.version}.`
                                     : ` Already current (${r.version}).`;
        // Re-read the checks so the row re-dates itself rather than sitting
        // there still flagged after the thing it flagged has been fixed.
        store.setDoctor(await api.doctor());
      } catch (e) {
        said.textContent = ` ${e.message || "That didn't work."}`;
      } finally {
        // Always live again. "Already current" leaves every value on the row
        // unchanged, so the panel does not re-render and the button would stay
        // greyed out for the rest of the session — on the one control somebody
        // may well want twice, since what makes it necessary is YouTube
        // changing something rather than anything happening in here.
        button.disabled = false;
      }
    };
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
