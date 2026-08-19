import { BaseElement } from "../base-element.js";
import { api } from "../api.js";

const SHELL = `
<dialog id="dlg" aria-labelledby="dlgTitle">
  <div class="modal-card">
    <div class="modal-head">
      <div class="modal-title" id="dlgTitle">Details to send
        <p class="hint">Paste this into a message to whoever helps you with this
          app. It describes your Mac and what the app has been doing. It contains
          no passwords or API keys.</p>
      </div>
      <button class="modal-close" id="xBtn" aria-label="Close"></button>
    </div>
    <div class="modal-body report">
      <pre id="diagText" tabindex="0">Gathering…</pre>
    </div>
    <div class="modal-foot">
      <button class="primary" id="copyBtn" autofocus>Copy</button>
      <button class="ghost" id="closeBtn">Close</button>
      <span id="diagMsg" class="hint" role="status" aria-live="polite"></span>
    </div>
  </div>
</dialog>
`;

const STYLE = `
<style>
  /* The report is the body's own surface, so the scroll shadow that shared.css
     masks with --panel has to be masked in this colour instead. */
  .modal-body.report{--panel:var(--bg);background-color:var(--bg);padding:0}
  #diagText{
    margin:0;max-height:none;padding:18px 20px;
    background:transparent;border:none;border-radius:0;
    font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
    font-size:13px;line-height:1.65;
    white-space:pre-wrap;overflow-wrap:anywhere}
  /* :focus, not :focus-visible — the clipboard fallback focuses this from a
     mouse click, and an outward ring would be clipped by the scroller. */
  #diagText:focus{outline:2px solid var(--accent);outline-offset:-3px}
  #diagMsg{margin:0;flex:1 1 200px;min-width:0}
  @media (max-width:420px){
    #diagText{padding:15px;font-size:12px}
  }
</style>
`;

class DiagnosticsPanel extends BaseElement {
  connectedCallback(){
    this.html(STYLE + SHELL);

    this.$("#copyBtn").onclick = () => this.copy();
    this.$("#closeBtn").onclick = () => this.close();
    this.$("#xBtn").onclick = () => this.close();

    const dlg = this.$("#dlg");
    // The press has to start on the backdrop as well as end there: the fallback
    // asks people to select the report, and a selection dragged past the card
    // would otherwise close the dialog out from under them.
    dlg.addEventListener("mousedown", e => { this._pressedBackdrop = e.target === dlg; });
    dlg.addEventListener("click", e => {
      if(e.target === dlg && this._pressedBackdrop) this.close();
    });
  }

  async open(){
    const dlg = this.$("#dlg");
    // Reachable from a failed job and from inside Settings, and showModal()
    // throws on a dialog that is already showing.
    if(!dlg.open) dlg.showModal();
    this.$("#diagMsg").textContent = "";
    this.$("#diagText").textContent = "Gathering…";
    try{
      const r = await api.diagnostics();
      this.$("#diagText").textContent = r.text;
    }catch(e){
      // Even this failing is worth something to send, so it goes in the box
      // rather than into a message that replaces the text with an apology.
      this.$("#diagText").textContent =
        "Could not gather the details (" + e + ").\nThe app may have stopped "
        + "responding. The log is at ~/Library/Logs/DubbingStudio.log";
    }
  }

  close(){ this.$("#dlg").close(); }

  async copy(){
    const btn = this.$("#copyBtn");
    try{
      await navigator.clipboard.writeText(this.$("#diagText").textContent);
      this.$("#diagMsg").textContent = "Copied — now paste it into a message.";
      // Only on success. Flipping the label regardless put "Copied ✓" next to
      // "couldn't reach the clipboard", and of the two the button is the one
      // people believe.
      btn.textContent = "Copied ✓";
      setTimeout(()=>{ btn.textContent = "Copy"; }, 2000);
    }catch(e){
      this.$("#diagMsg").textContent = "Couldn't reach the clipboard — select the text "
        + "above and copy it with ⌘C.";
      this.$("#diagText").focus();
    }
  }
}

customElements.define("diagnostics-panel", DiagnosticsPanel);
