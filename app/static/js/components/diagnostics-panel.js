import { BaseElement } from "../base-element.js";
import { api } from "../api.js";

const SHELL = `
<dialog id="dlg">
  <div class="panel" style="margin:0;max-width:560px;width:calc(100vw - 48px)">
    <p class="job-title">Details to send</p>
    <p class="hint" style="margin-top:6px">Paste this into a message to whoever
      helps you with this app. It describes your Mac and what the app has been
      doing. It contains no passwords or API keys.</p>
    <pre id="diagText" tabindex="0" style="max-height:320px;overflow:auto;font-size:11px;
         line-height:1.5;white-space:pre-wrap;word-break:break-word;
         background:var(--accent-soft);border:1px solid var(--line);
         border-radius:10px;padding:12px;margin-top:12px">Gathering…</pre>
    <div style="margin-top:16px;display:flex;gap:8px;align-items:center">
      <button class="primary" id="copyBtn">Copy</button>
      <button class="ghost" id="closeBtn">Close</button>
      <span id="diagMsg" class="hint" style="margin:0" role="status"
            aria-live="polite"></span>
    </div>
  </div>
</dialog>
`;

class DiagnosticsPanel extends BaseElement {
  connectedCallback(){
    this.html(`<style>
      dialog{border:none;border-radius:var(--radius);padding:0;background:transparent;
             max-width:560px;width:calc(100vw - 48px)}
      dialog::backdrop{background:rgba(0,0,0,.4)}
    </style>` + SHELL);

    this.$("#copyBtn").onclick = () => this.copy();
    this.$("#closeBtn").onclick = () => this.close();
  }

  async open(){
    this.$("#dlg").showModal();
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
