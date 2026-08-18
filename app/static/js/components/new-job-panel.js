import { BaseElement } from "../base-element.js";
import { store } from "../store.js";
import { escapeHtml, escapeAttr } from "../format.js";

const SHELL = `
  <div class="panel">
    <label for="url">Video link</label>
    <div class="row">
      <input type="text" id="url" placeholder="https://www.youtube.com/watch?v=…"
             autocomplete="off" spellcheck="false">
      <button class="ghost" id="tryBtn" data-busy="start">Try 30 seconds</button>
      <button class="primary" id="go" data-busy="start">Dub it</button>
    </div>
    <p class="hint"><b>Try 30 seconds</b> dubs a short sample first, so you
      can hear the voice before waiting for a whole video.</p>

    <div class="preset-wrap">
      <div class="segmented" id="presets"></div>
      <p class="hint" id="presetBlurb"></p>
      <p class="hint hidden" id="presetBlocked" role="status" aria-live="polite"></p>

      <div class="grid" style="margin-top:16px">
        <div>
          <label for="glossary">What kind of video is this?</label>
          <select id="glossary"></select>
        </div>
        <div>
          <label for="diarize">Who's speaking?</label>
          <select id="diarize">
            <option value="false">One person</option>
            <option value="true">Several people</option>
          </select>
        </div>
      </div>
      <p class="hint" id="glossaryHint" role="status" aria-live="polite"></p>
      <p class="hint" id="speakersHint" role="status" aria-live="polite"></p>
    </div>
  </div>
`;

function blockedReasons(features){
  const f = features || {};
  return {
    best: (!f.cloning || !f.whisper) ? "Needs the quality extras — see Setup check" : "",
    balanced: !f.separation ? "Music separation isn't installed" : "",
  };
}

function glossaryHint(value){
  return value && value !== "none"
    ? "Specialist terms will be translated the way this subject uses them, not "
      + "the way a dictionary would. The app also learns the video's own "
      + "vocabulary as it goes, whatever language it is in."
    : "The app works out a video's specialist vocabulary by itself, in any "
      + "language. Naming the subject helps it choose between meanings, and "
      + "for crochet it decides whether to use US or UK stitch names.";
}

function speakersHint(value){
  return value === "true"
    ? "Each speaker gets their own voice. Telling people apart is the least "
      + "reliable step, so check the result — if it finds more people than are "
      + "really there, say how many in Settings."
    : "Faster, and it can't mistake one person for several — which is the most "
      + "common way a dub of a tutorial goes wrong.";
}

class NewJobPanel extends BaseElement {
  connectedCallback(){
    this.html(SHELL);
    this.$("#tryBtn").onclick = () => this.start(true);
    this.$("#go").onclick = () => this.start(false);
    this.$("#url").addEventListener("keydown", e => { if(e.key === "Enter") this.start(false); });
    this.$("#glossary").onchange = e =>
      this.emit("save-settings", {data: {glossary: e.target.value}});
    this.$("#diarize").onchange = e =>
      this.emit("save-settings", {data: {diarize: e.target.value === "true"}});
    this._unsub = store.subscribe(s => this.update(s));
  }

  disconnectedCallback(){
    this._unsub?.();
  }

  setUrl(value){
    this.$("#url").value = value;
  }

  focusUrl(){
    this.$("#url").focus();
  }

  start(preview){
    const url = this.$("#url").value.trim();
    if(!url) return;
    this.emit("start-job", {url, preview: !!preview});
  }

  update(s){
    const { settings, presets, glossaries, features } = s;
    this.renderIfChanged(
      [settings.preset, settings.glossary, settings.diarize, features],
      () => {
        const blocked = blockedReasons(features);
        this.$("#presets").innerHTML = Object.entries(presets || {}).map(([k,p]) =>
          `<button class="${k===settings.preset?"on":""}" data-preset="${k}"
                   ${blocked[k]?`disabled title="${escapeAttr(blocked[k])}"`:""}
                   >${escapeHtml(p.label)}</button>`).join("");
        this.$$("#presets button").forEach(b => {
          b.onclick = () => this.emit("save-settings", {data: {preset: b.dataset.preset}});
        });
        // "custom" is a real state now that the preset is read back off the
        // switches rather than left as whatever was last named.
        const p = (presets || {})[settings.preset];
        this.$("#presetBlurb").textContent = blocked[settings.preset] || (p ? p.blurb
          : "Your own mix of settings — change them back in Settings to return to one "
            + "of these.");
        const notes = Object.entries(blocked).filter(([,why]) => why)
          .map(([k,why]) => `${presets[k]?.label || k}: ${why}`);
        this.$("#presetBlocked").textContent = notes.join(". ");
        this.$("#presetBlocked").classList.toggle("hidden", !notes.length);

        this.$("#glossary").innerHTML = Object.entries(glossaries || {})
          .map(([k,l])=>`<option value="${escapeAttr(k)}">${escapeHtml(l)}</option>`).join("");
        this.$("#glossary").value = settings.glossary ?? "";
        this.$("#diarize").value = String(!!settings.diarize);
        this.$("#glossaryHint").textContent = glossaryHint(settings.glossary);
        this.$("#speakersHint").textContent = speakersHint(String(!!settings.diarize));
      }
    );
  }
}

customElements.define("new-job-panel", NewJobPanel);
