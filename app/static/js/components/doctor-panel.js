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
    // What the yt-dlp row is currently saying, held here rather than only on
    // the node. Refreshing the checks after an update re-renders the whole
    // list, so a message written straight into the DOM was wiped by the very
    // refresh that proved it had worked — and only "Already current" survived,
    // because that is the case where nothing changes and nothing re-renders.
    this._ytdlp = {said: "", busy: false};
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
      // The message belongs to the update somebody just ran, not to the panel.
      // It has to outlive the re-render that update causes — that re-render is
      // what used to wipe it — and it must not outlive anything else, or a
      // refresh an hour later paints " Updated to 2026.08.19." beside the
      // button as though it had just happened.
      //
      // `busy` is the whole mechanism. It is still set while the refresh below
      // runs, so that render keeps the message; it is cleared immediately
      // after, so the next render — which belongs to something else — drops it.
      // Cleared here rather than after the markup, because the markup is what
      // reads it: deciding afterwards still painted it this time round and only
      // took effect on the render after that.
      if(!this._ytdlp.busy && this._ytdlp.said) this._ytdlp = {said: "", busy: false};
      this.$("#doctor").innerHTML = s.doctor.checks.map(c=>`
        <div class="check">
          <span class="dot ${c.ok?"ok":(c.optional?"opt":"bad")}"></span>
          <div><b>${escapeHtml(c.name)}</b>${c.optional&&!c.ok?'<span class="tag">optional</span>':""}
            ${c.ok?"":`<br><code>${escapeHtml(c.hint)}</code>`}
            ${c.note?`<br><code>${escapeHtml(c.note)}</code>`:""}
            ${c.action==="update-ytdlp"?`<br><button class="small" data-action="update-ytdlp"
                 style="margin-top:6px"${this._ytdlp.busy?" disabled":""}>Update yt-dlp</button>
               <span class="hint" data-said="update-ytdlp">${
                 escapeHtml(this._ytdlp.said)}</span>`:""}</div>
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
    button.onclick = async () => {
      this.sayYtdlp(" Updating…", true);
      let done;
      try {
        const r = await api.updateYtdlp();
        // Already current is a success, and saying so beats a silent button:
        // somebody pressing this is trying to fix a download that just failed,
        // and needs to know whether to look elsewhere.
        done = r.changed ? ` Updated to ${r.version}.`
                         : ` Already current (${r.version}).`;
      } catch (e) {
        // Painted here and not held: nothing re-renders behind a failure, so
        // this is the only thing that shows it, and clearing busy is what stops
        // it reappearing beside the button an hour later.
        this.sayYtdlp(` ${e.message || "That didn't work."}`, false);
        return;
      }
      // Said now, but still held: what the button starts is a pip install into
      // the environment this app is running out of, and a second one racing the
      // first writes over the same files. Staying disabled until everything it
      // set off has finished is what makes a double-click harmless from here.
      this.sayYtdlp(done, true);
      try {
        // Re-read the checks so the row re-dates itself rather than sitting
        // there still flagged after the thing it flagged has been fixed.
        store.setDoctor(await api.doctor());
      } catch {
        // Deliberately silent, and deliberately after the message above: the
        // update itself succeeded, and letting a failed refresh replace that
        // with "the request failed" would report the wrong outcome. The row
        // stays as it was and the next refresh corrects it.
      }
      // Only now: this gives the button back and, by clearing busy, marks the
      // message as belonging to a run that has finished, so the next render
      // that is not this one's drops it.
      this.sayYtdlp(done, false);
    };
  }

  // Records what the row says and paints it at once. Both halves matter: the
  // paint shows it immediately even when nothing re-renders, and the record is
  // what puts it back when something does. Whether it survives the next render
  // is decided by `busy` — see update().
  sayYtdlp(said, busy){
    this._ytdlp = {said, busy};
    const button = this.$('[data-action="update-ytdlp"]');
    const span = this.$('[data-said="update-ytdlp"]');
    if(button) button.disabled = busy;
    if(span) span.textContent = said;
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
