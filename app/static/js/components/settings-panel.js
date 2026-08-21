import { BaseElement } from "../base-element.js";
import { store } from "../store.js";
import { api } from "../api.js";
import { escapeAttr, escapeHtml, flash } from "../format.js";
import "./info-tip.js";

const BOOL_SETTINGS = ["write_srt","separate_audio","diarize","keep_music","merge_lines"];
const NUM_SETTINGS = ["speed","max_stretch"];
const TEXT_SETTINGS = ["voice","glossary","translator","ollama_model",
  "youtube_cookies",
  "anthropic_key","openai_key","anthropic_model","openai_model",
  "custom_glossary","keep_video_quality","asr_model","voice_mode"];

// One control for two settings. duck_db is read in duck mode and nowhere else —
// mux.mux() consults it only there — so there is no separate question to ask,
// and each option value carries the mode plus, where it applies, the level the
// original is held at. audio_mode is therefore painted and posted by hand
// rather than through the lists above, and names duck_db in its data-also so
// the setting it carries is not invisible to the reset scopes.
//
// The dB figure is the only thing telling the three duck options apart, and
// a closed <select> shows only as much of its chosen option as the control is
// wide — spelling out "the whole original" here once pushed the figure past
// that edge, so the selection itself could no longer be read without opening
// the menu. That is a worse failure than the misreading it was trying to
// fix. The labels stay short; what they mean — that duck and dual both keep
// the whole original, its own speech included, not only the music riding
// along in it — is said once, in full, in the hint under the control instead.
const AUDIO_CHOICES = [
  ["replace", "Replace completely"],
  ["duck:-12", "Keep quietly underneath — quiet (-12 dB)"],
  ["duck:-18", "Keep quietly underneath — very quiet (-18 dB)"],
  ["duck:-24", "Keep quietly underneath — barely there (-24 dB)"],
  ["dual", "Keep as a second track"],
];

// The tabs, in order, and the only place they are declared: the strip is built
// from this, the panes are shown and hidden by their data-pane, and what each
// tab owns is read off its pane. See fields().
const TABS = [
  {key: "voice", label: "Voice & Audio"},
  {key: "translation", label: "Translation"},
  {key: "advanced", label: "Advanced"},
];

const PRESET_TAG = `<span class="tag">Set by preset</span>`;

const AUDIO_OPTIONS = AUDIO_CHOICES.map(([value, label]) =>
  `<option value="${escapeAttr(value)}">${escapeHtml(label)}</option>`).join("");

const SHELL = `
<dialog id="dlg" aria-labelledby="dlgTitle">
  <div class="modal-card">
    <div class="modal-head">
      <div class="modal-title" id="dlgTitle">Settings
        <p class="hint" id="dlgSummary"></p>
      </div>
      <button class="modal-close" id="xBtn" aria-label="Close"></button>
    </div>

    <div class="modal-body">
      <div class="segmented" id="settingsTabs"></div>
      <p class="hint" id="presetLine">Three settings here are the ones a quality
        preset chooses, tagged <b>Set by preset</b>. Changing one by hand turns the
        preset to Custom. A preset touches nothing else in this window.</p>

      <div data-pane="voice">
        <div class="grid">
          <div>
            <div class="field-head"><label for="voice">Voice</label></div>
            <select id="voice"></select>
            <div style="display:flex;gap:8px;align-items:center;margin-top:8px">
              <button class="ghost" id="auditionBtn">Hear this voice</button>
              <span class="hint" id="auditionMsg" style="margin:0" role="status" aria-live="polite"></span>
            </div>
            <audio id="auditionAudio" style="display:none"></audio>
          </div>
          <!-- Named "Cloning" rather than "Voices", which reads too easily as
               a duplicate of "Voice" beside it — "Cloning" is the word the
               rest of the app (the info-tip below, new-job-panel.js) already
               uses for this same choice. -->
          <div>
            <div class="field-head"><label for="voice_mode">Cloning</label>${PRESET_TAG}<info-tip
              label="cloning"
              text="Cloning keeps the speaker's identity, but carries their accent into English."></info-tip></div>
            <select id="voice_mode">
              <option value="fixed">Use a built-in voice</option>
              <option value="clone">Clone the original speaker</option>
            </select>
          </div>
          <!-- Full width, not a half column: three of the five options carry a
               dB figure that is the only thing telling them apart, and a half
               column clips it off the end. -->
          <div class="grid-full">
            <div class="field-head"><label for="audio_mode">Original audio</label></div>
            <select id="audio_mode" data-also="duck_db" aria-describedby="audioModeHint">${AUDIO_OPTIONS}</select>
            <p class="hint hidden" id="audioModeHint"></p>
          </div>
          <div>
            <div class="field-head"><label for="separate_audio">Music and effects</label>${PRESET_TAG}<info-tip
              label="music and effects"
              text="Splits speech from the rest so replacing the voices doesn't wipe the soundtrack."></info-tip></div>
            <select id="separate_audio">
              <option value="true">Separate and keep them</option>
              <option value="false">Don't separate (faster)</option>
            </select>
          </div>
          <div id="keepMusicBox">
            <div class="field-head"><label for="keep_music">The separated music and effects</label><span
              class="tag off hidden" id="keepMusicTag" data-override="keep_music">Not in force</span></div>
            <select id="keep_music" aria-describedby="keepMusicTag keepMusicHint">
              <option value="true">Mix it back under the new voices</option>
              <option value="false">Drop it — voices only</option>
            </select>
            <p class="hint hidden" id="keepMusicHint"></p>
          </div>
          <div>
            <div class="field-head"><label for="speed">Speaking speed</label></div>
            <select id="speed">
              <option value="0.9">Slower</option>
              <option value="1.0">Normal</option>
              <option value="1.1">Slightly faster</option>
            </select>
          </div>
        </div>
      </div>

      <div data-pane="translation">
        <div class="grid">
          <div>
            <div class="field-head"><label for="translator">Translated by</label></div>
            <select id="translator">
              <option value="ollama">Local model (free, private)</option>
              <option value="anthropic">Claude API (best quality)</option>
              <option value="openai">OpenAI API</option>
            </select>
          </div>
          <div id="languageBox" data-also="target_language"></div>
        </div>
        <div class="grid" style="margin-top:14px">
          <div id="ollamaBox">
            <div class="field-head"><label for="ollama_model">Local model</label><info-tip
              id="modelTip" class="hidden" label="the local model"></info-tip></div>
            <input type="text" id="ollama_model" placeholder="auto">
          </div>
          <div id="anthropicBox" class="hidden">
            <div class="field-head"><label for="anthropic_key">Anthropic API key</label></div>
            <input type="password" id="anthropic_key" data-noreset placeholder="sk-ant-…">
            <div class="field-head" style="margin-top:10px"><label
              for="anthropic_model">Claude model</label></div>
            <input type="text" id="anthropic_model" placeholder="claude-sonnet-5">
            <p class="hint">The key is saved in plain text in
              <code id="settingsPath1">settings.json</code> on this computer, so that
              the app can translate without asking for it again. Anyone with access to
              your account can read it.</p>
          </div>
          <div id="openaiBox" class="hidden">
            <div class="field-head"><label for="openai_key">OpenAI API key</label></div>
            <input type="password" id="openai_key" data-noreset placeholder="sk-…">
            <div class="field-head" style="margin-top:10px"><label
              for="openai_model">OpenAI model</label></div>
            <input type="text" id="openai_model" placeholder="gpt-4o">
            <p class="hint">The key is saved in plain text in
              <code id="settingsPath2">settings.json</code> on this computer, so that
              the app can translate without asking for it again. Anyone with access to
              your account can read it.</p>
          </div>
        </div>
        <div style="margin-top:14px">
          <div class="field-head"><label for="glossary">Crochet stitch names</label></div>
          <select id="glossary"></select>
        </div>
        <div style="margin-top:14px">
          <div class="field-head"><label for="custom_glossary">Your own terms (one per
            line, “as spoken → English”)</label></div>
          <textarea id="custom_glossary" placeholder="ponto amêndoa -> almond stitch"></textarea>
          <p class="hint">Used together with the stitch names above. The rest of a
            video's specialist vocabulary the app works out from the video itself, in
            any language.</p>
        </div>
      </div>

      <div data-pane="advanced">
        <div class="grid">
          <div>
            <div class="field-head"><label for="asr_model">Transcription
              engine</label>${PRESET_TAG}</div>
            <select id="asr_model">
              <option value="parakeet">Parakeet — fast</option>
              <option value="whisper">Whisper — more accurate, slower</option>
            </select>
          </div>
          <div>
            <div class="field-head"><label for="youtube_cookies">Sign in as</label><info-tip
              label="signing in"
              text="For videos YouTube refuses to send to a signed-out request. Borrows the session from a browser you're already signed into on this Mac. Nothing is uploaded; the cookies are only sent to the site the video is on."></info-tip></div>
            <select id="youtube_cookies">
              <option value="">Don't sign in</option>
              <option value="safari">Safari</option>
              <option value="chrome">Chrome</option>
              <option value="firefox">Firefox</option>
              <option value="edge">Edge</option>
              <option value="brave">Brave</option>
            </select>
          </div>
          <div>
            <div class="field-head"><label for="keep_video_quality">Video
              quality</label><info-tip label="video quality"
              text="The picture is copied, never re-encoded, so this decides the download size. On an hour-and-a-half tutorial, 1080p is around 1.8 GB and 720p around 730 MB — and 720p is plenty for following along."></info-tip></div>
            <select id="keep_video_quality">
              <option value="best">Best available</option>
              <option value="1080">Up to 1080p</option>
              <option value="720">Up to 720p — much smaller</option>
            </select>
          </div>
          <div>
            <div class="field-head"><label for="write_srt">Subtitles</label></div>
            <select id="write_srt">
              <option value="false">No subtitle file</option>
              <option value="true">Also save an .srt</option>
            </select>
          </div>
          <div>
            <div class="field-head"><label for="merge_lines">Run-on lines</label><info-tip
              label="run-on lines"
              text="Fast dialogue arrives as many very short lines with no gap between them, and each has to be squeezed to fit. Joining them gives the translation room. Material with real pauses is unaffected."></info-tip></div>
            <select id="merge_lines">
              <option value="true">Join lines that run together</option>
              <option value="false">Keep them exactly as heard</option>
            </select>
          </div>
          <div>
            <div class="field-head"><label for="max_stretch">Hardest allowed
              squeeze</label><info-tip label="the hardest allowed squeeze"
              text="How much a line may be sped up to fit the gap the original speaker left. Past about 1.6x it starts to sound hurried; beyond the limit the line runs on and later pauses absorb it."></info-tip></div>
            <select id="max_stretch">
              <option value="1.3">Gentle — 1.3x</option>
              <option value="1.55">Normal — 1.55x</option>
              <option value="1.8">Firm — 1.8x</option>
            </select>
          </div>
        </div>
      </div>

      <div class="danger-row">
        <div>
          <b>Restore defaults</b>
          <small>Puts settings back to what the app ships with, straight away — there
            is nothing to save afterwards. Finished videos and saved API keys are
            never touched.</small>
        </div>
        <button class="ghost icon-btn" id="resetTabBtn" data-busy="reset">Reset this tab</button>
        <button class="ghost icon-btn" id="resetAllBtn" data-busy="reset">Reset everything</button>
      </div>

      <div class="danger-row">
        <div>
          <b>Details to send</b>
          <small>Describes this Mac and what the app has been doing, for when you need
            to ask someone about it. No passwords or API keys.</small>
        </div>
        <button class="ghost icon-btn" id="diagBtn">Copy details</button>
      </div>
    </div>

    <div class="modal-foot">
      <button class="primary" id="saveBtn" data-busy="save">Save</button>
      <button class="ghost" id="closeBtn">Close</button>
      <span id="savedMsg" class="hint" style="margin:0" role="status" aria-live="polite"></span>
    </div>
  </div>
</dialog>
`;

const STYLE = `
<style>
  #presetLine{margin:10px 0 18px}
  /* The label, the preset tag, the info-tip button and the changed flag sit in
     a row of their own. Anything inside a <label> joins the accessible name of
     the control it labels, which turned one field into four concepts read as
     its name. */
  .field-head{display:flex;align-items:center;flex-wrap:wrap;margin-bottom:6px}
  .field-head label{margin-bottom:0}
  /* A value with nothing to choose between is stated rather than drawn: both a
     one-item menu and a disabled one read as something that ought to work. */
  .stated{margin:0;font-size:15px;color:var(--ink)}
  /* Ambient, not an alarm: a word rather than a colour, so it still says what
     it says to someone who cannot see the difference. The count on a tab is the
     same pill, tightened up and taking the button's own colour so it dims with
     it. Letter-spacing and case are normalised because both sit inside type
     that sets them — a label, a segmented button. */
  .pill{margin-left:6px;padding:1px 7px;font-size:10.5px;font-weight:600;
        border:1px solid var(--line);border-radius:99px;color:var(--muted);
        vertical-align:middle;text-transform:none;letter-spacing:.01em}
  .pill.count{padding:0 5px;font-weight:700;font-variant-numeric:tabular-nums;
              border-color:currentColor;color:inherit;opacity:.7}
  .pill:empty{display:none}
  .pill.flag:not(.on){display:none}
  /* "Set by preset" is drawn in the accent colour because it is telling the
     truth about a control that is fully in force. A control this same tag
     shape sits on that has been quietly overridden needs the opposite
     read at a glance — dim rather than featured — so the same silhouette
     does not say the same thing twice over for two opposite states. */
  .tag.off{background:transparent;border:1px solid var(--line);color:var(--muted)}
  /* De-emphasis that keeps the control operable and in the tab order, unlike
     the native disabled state this replaced: a disabled select is unreachable
     by keyboard, which hid the one field this item exists to make legible
     from exactly the person who cannot see a greyed-out control change
     colour. Opacity alone dims every part of it together — text, border,
     arrow — without touching focusability. */
  select.dim{opacity:.65}
  /* Spans both tracks of the two-column grid it sits in, rather than sharing
     one — see the markup comment above #audio_mode for why a half column
     wasn't enough room for that control. At the single-column width the media
     query below already forces, this is a no-op: there is only one track to
     span. */
  .grid-full{grid-column:1/-1}
  /* A save that did not happen must not read as the acknowledgement it stands
     in the place of. */
  #savedMsg.bad{color:var(--bad)}
</style>`;

function same(a, b){
  if(typeof a === "boolean" || typeof b === "boolean") return !!a === !!b;
  const na = Number(a), nb = Number(b);
  if(a !== "" && b !== "" && Number.isFinite(na) && Number.isFinite(nb)) return na === nb;
  return String(a ?? "") === String(b ?? "");
}

class SettingsPanel extends BaseElement {
  connectedCallback(){
    this.html(STYLE + SHELL);

    this._tab = TABS[0].key;
    this._dirty = new Set();

    this.$("#settingsTabs").innerHTML = TABS.map((t, i) =>
      `<button data-tab="${escapeAttr(t.key)}"${i ? "" : " autofocus"}><span class="mark"
        aria-hidden="true">✓</span>${escapeHtml(t.label)}<span
        class="pill count" data-tabflag="${escapeAttr(t.key)}" aria-hidden="true"></span></button>`).join("");

    this.$("#saveBtn").onclick = () => this.save();
    this.$("#closeBtn").onclick = () => this.close();
    this.$("#xBtn").onclick = () => this.close();
    this.$("#auditionBtn").onclick = () => this.audition();
    this.$("#resetTabBtn").onclick = e => this.resetTab(e.currentTarget);
    this.$("#resetAllBtn").onclick = e => this.resetAll(e.currentTarget);
    this.$("#diagBtn").onclick = () => this.showDiagnostics();
    this.$$("#settingsTabs button").forEach(b => {
      b.onclick = () => this.selectTab(b.dataset.tab);
    });

    // The three controls another field's presence depends on. Hiding one takes
    // it out of the counts as well, which is why they are recounted here and
    // not only when the store says something changed.
    ["translator", "separate_audio", "audio_mode"].forEach(id => {
      this.$(`#${id}`).addEventListener("change", () => {
        this.applyConditions();
        this.refreshCounts();
      });
    });

    this.watchEdits();
    this.selectTab(this._tab);

    const dlg = this.$("#dlg");
    dlg.addEventListener("pointerdown", e => { this._fromBackdrop = e.target === dlg; });
    dlg.addEventListener("click", e => {
      if(e.target === dlg && this._fromBackdrop) dlg.close();
    });

    this._unsub = store.subscribe(s => this.update(s));
  }

  disconnectedCallback(){
    this._unsub?.();
  }

  // The saved value is what a field is painted from, so an edit that has not
  // been saved yet has to be remembered, or the next thing to write settings
  // — a reset, or the toggle in the header — repaints over the top of it.
  watchEdits(){
    this.$$("input, select, textarea").forEach(el => {
      if(!el.id) return;
      const mark = () => this._dirty.add(el.id);
      el.addEventListener("input", mark);
      el.addEventListener("change", mark);
    });
  }

  placeFlags(){
    TABS.forEach(tab => this.fields(tab).forEach(key => {
      const head = this.$(`label[for="${key}"]`)?.closest(".field-head");
      if(!head || head.querySelector(`[data-flag="${key}"]`)) return;
      const flag = document.createElement("span");
      flag.className = "pill flag";
      flag.dataset.flag = key;
      flag.textContent = "changed";
      head.append(" ", flag);
    }));
  }

  // What a tab owns, read off its pane rather than written down a second time:
  // an id naming one of the shipped defaults is a setting, which leaves the
  // buttons, boxes and file paths in the panes out of it. data-also names a
  // setting with no control of its own — duck_db, chosen by the audio menu
  // along with the mode, and target_language, which the voices decide.
  // data-noreset marks a field no reset may name: the server only clears an API
  // key when it is asked for it by name, because a key is pasted in from an
  // account elsewhere and cannot be read back out of this app, and emptying the
  // box and pressing Save still removes one.
  //
  // The list is the reset scope: "Reset this tab" puts all of it back, including
  // a setting whose control is off screen at the time, because a value left at
  // something other than the shipped one is exactly what the button promises to
  // undo. The counts — the badge on the tab, and the number in the reset prompt
  // — are taken over the fields the user can actually see instead, so neither of
  // them ever points at something invisible. See shown().
  fields(tab){
    if(!this._defaults) return [];
    const cached = (this._fields ??= {})[tab.key];
    if(cached) return cached;
    const keys = [];
    this.$(`[data-pane="${tab.key}"]`).querySelectorAll("[id], [data-also]")
      .forEach(el => {
        if(Object.hasOwn(this._defaults, el.id) && !("noreset" in el.dataset)) keys.push(el.id);
        if(el.dataset.also) keys.push(el.dataset.also);
      });
    return (this._fields[tab.key] = keys);
  }

  open(){
    this.note("", 0);
    this.$("#dlg").showModal();
    // Chrome will focus the scrolling body otherwise, and ring the whole of it.
    this.$("#settingsTabs button.on")?.focus();
  }

  close(){ this.$("#dlg").close(); }

  tab(){ return TABS.find(t => t.key === this._tab) || TABS[0]; }

  selectTab(key){
    this._tab = key;
    this.$$("#settingsTabs button").forEach(b => {
      const on = b.dataset.tab === key;
      b.classList.toggle("on", on);
      b.setAttribute("aria-pressed", String(on));
    });
    this.$$("[data-pane]").forEach(p => p.classList.toggle("hidden", p.dataset.pane !== key));
    this.$("#resetTabBtn").textContent = `Reset ${this.tab().label}`;
  }

  // Nothing here is touched by a job, and the store notifies about twice a
  // second for as long as one runs, into a dialog that is usually closed. The
  // key is every value the paint reads.
  update(s){
    const { settings, settings_defaults, voices, glossaries, machine, settings_path } = s;
    if(!Object.keys(settings).length) return;
    this.renderIfChanged(
      [settings, settings_defaults, voices, glossaries, settings_path,
       machine?.suggested_model, machine?.ram_gb],
      () => this.repaint(s));
  }

  repaint(s){
    const { settings, settings_defaults, voices, glossaries, machine, settings_path } = s;
    this._state = s;
    this._settings = settings;
    this._defaults = settings_defaults || {};

    this.fillOptions("#voice", (voices || []).map(o => [o.id, o.label]));
    this.fillOptions("#glossary", Object.entries(glossaries || {}));
    this.languageControl(voices);
    this.placeFlags();

    TEXT_SETTINGS.forEach(k => this.paint(k, el => { el.value = settings[k] ?? ""; }));
    BOOL_SETTINGS.forEach(k => this.paint(k, el => { el.value = String(!!settings[k]); }));
    // Matched numerically, not as a string. JSON turns Python's 1.0 into JS 1,
    // and String(1) is "1", which matches none of the "0.9"/"1.0"/"1.1" options.
    NUM_SETTINGS.forEach(k => this.paint(k, el => {
      const want = Number(settings[k] ?? this._defaults[k]);
      const opt = el.tagName === "SELECT"
        ? [...el.options].find(o => Number(o.value) === want) : null;
      el.value = opt ? opt.value : (el.tagName === "SELECT" ? el.value : String(want));
    }));
    this.paintAudio(settings);

    this.applyConditions();
    this.markChanged(settings, this._defaults);

    this.$("#settingsPath1").textContent = settings_path || "settings.json";
    this.$("#settingsPath2").textContent = settings_path || "settings.json";

    if(machine && machine.suggested_model){
      const tip = this.$("#modelTip");
      tip.text =
        `Leave blank to use ${machine.suggested_model}, chosen for your `
        + `${machine.ram_gb} GB of memory. You can name any model Ollama has — `
        + `qwen3:32b translates specialist material better, but it is a 20 GB `
        + `download and slower per line. Pull it in Ollama first; if it isn't there, `
        + `the nearest installed model is used instead and the report says so.`;
      tip.classList.remove("hidden");
    }
  }

  // Rebuilt only when the list itself changes: writing the options again resets
  // the selection, which would throw away an unsaved edit.
  fillOptions(selector, pairs){
    const el = this.$(selector);
    if(!el) return;
    const key = pairs.map(([value]) => value).join(",");
    if(el.dataset.options === key) return;
    el.dataset.options = key;
    el.innerHTML = pairs.map(([value, label]) =>
      `<option value="${escapeAttr(value)}">${escapeHtml(label)}</option>`).join("");
  }

  // The languages on offer are read off the voice inventory, because only a
  // voice can speak a translation: taking them from there is what stops a
  // language being offered that nothing could then say out loud. Every voice the
  // app ships is an English one, so there is nothing to choose — the value is
  // stated rather than drawn as a control that cannot do anything, and no
  // control carries it, since what would be saved is what the inventory already
  // says. A voice in another language wants a real menu here again.
  languageControl(voices){
    const langs = [...new Set((voices || []).map(v => v.language).filter(Boolean))];
    if(!langs.length) return;
    if(this._language === langs[0]) return;
    this._language = langs[0];
    const lang = escapeHtml(langs[0]);
    this.$("#languageBox").innerHTML =
      `<div class="field-head"><label>Translate into</label></div>
       <p class="stated">${lang}</p>
       <p class="hint">Every voice the app has speaks ${lang}, so that is the
         only language it can dub into.</p>`;
  }

  paint(key, write){
    const el = this.$(`#${key}`);
    if(el && !this._dirty.has(key)) write(el);
  }

  paintAudio(settings){
    this.paint("audio_mode", el => {
      const mode = settings.audio_mode;
      if(mode !== "duck"){
        if([...el.options].some(o => o.value === mode)) el.value = mode;
        return;
      }
      // A level that is none of the three on offer — hand-edited, or left over
      // from a build that offered others — takes the nearest one rather than
      // blanking the control or reading as the loudest.
      const want = Number(settings.duck_db);
      const level = o => Math.abs(Number(o.value.split(":")[1]) - want);
      el.value = [...el.options].filter(o => o.value.startsWith("duck:"))
        .reduce((a, b) => level(b) < level(a) ? b : a).value;
    });
  }

  // Whether a field's own control is on screen. A conditional wrapper closed
  // around it takes the field with it; a tab pane that simply is not the one on
  // top does not, because the counts are per tab. A setting with no control of
  // its own — duck_db and target_language, see fields() — is never counted, and
  // is never flagged either.
  shown(key){
    const el = this.$(`#${key}`);
    if(!el) return false;
    for(let node = el; node; node = node.parentElement){
      if(node.dataset.pane) return true;
      if(node.classList.contains("hidden")) return false;
    }
    return true;
  }

  visibleFields(tab){
    return this.fields(tab).filter(key => this.shown(key));
  }

  // Whether a field's own value currently has no bearing on the run — read
  // off the same tag applyConditions() shows beside it, the way shown() reads
  // the .hidden class it toggles, rather than re-deriving keep_music_applies()
  // a second time here in this file's own image. A field with no such tag
  // present is never overridden.
  overridden(key){
    const tag = this.$(`[data-override="${key}"]`);
    return !!tag && !tag.classList.contains("hidden");
  }

  refreshCounts(){
    if(this._settings) this.markChanged(this._settings, this._defaults || {});
  }

  // A control an override has made inert is not "changed" in any sense worth
  // reporting: the value on screen may be exactly what was saved, but with no
  // bearing on the run right now, counting it — or badging it "changed" right
  // beside its own "Not in force" tag — tells the user two contradictory
  // things about the same field in the same breath. See overridden().
  markChanged(settings, defaults){
    let total = 0;
    TABS.forEach(tab => {
      let count = 0;
      this.fields(tab).forEach(key => {
        const differs = key in defaults && !same(settings[key], defaults[key])
                        && this.shown(key) && !this.overridden(key);
        if(differs) count++;
        const flag = this.$(`[data-flag="${key}"]`);
        if(!flag) return;
        flag.classList.toggle("on", differs);
        if(differs) flag.title = `Ships as ${this.describe(key, defaults[key])}`;
      });
      const badge = this.$(`[data-tabflag="${tab.key}"]`);
      badge.textContent = count ? String(count) : "";
      const button = badge.closest("button");
      if(count) button.title = `${count} changed from what the app ships with`;
      else button.removeAttribute("title");
      total += count;
    });
    this.$("#dlgSummary").textContent = total
      ? `${total} setting${total === 1 ? "" : "s"} differ${total === 1 ? "s" : ""} from `
        + `what the app ships with, marked changed below.`
      : "Everything here is as the app ships.";
  }

  describe(key, value){
    const el = this.$(`#${key}`);
    if(el && el.tagName === "SELECT"){
      // An option's value is always a string, and same() reads a boolean on
      // either side as a truth test — under which "false" is true. A boolean is
      // therefore matched as the word the option carries.
      const want = typeof value === "boolean" ? String(value) : value;
      const opt = [...el.options].find(o => same(o.value, want));
      if(opt) return `“${opt.textContent.trim()}”`;
    }
    return value === "" || value == null ? "blank" : `“${value}”`;
  }

  applyConditions(){
    const t = this.$("#translator").value;
    this.$("#ollamaBox").classList.toggle("hidden", t !== "ollama");
    this.$("#anthropicBox").classList.toggle("hidden", t !== "anthropic");
    this.$("#openaiBox").classList.toggle("hidden", t !== "openai");

    // Both "duck" and "dual" carry the whole original along, its own speech
    // included, not only the music riding in it — easy to miss since the
    // option labels only say "keep it underneath" or "as a second track", so
    // the full explanation is spelled out here instead of in the label, where
    // it would push the dB figure that tells the three duck levels apart past
    // what a closed <select> shows.
    const replacing = this.$("#audio_mode").value.split(":")[0] === "replace";
    const audioHint = this.$("#audioModeHint");
    audioHint.classList.toggle("hidden", replacing);
    audioHint.textContent = replacing ? "" :
      "Keeps the whole original in the file, not just its music — the "
      + "original speech rides along with it.";

    // Mirrors Settings.keep_music_applies() in config.py: there's a separated
    // bed to put back only when speech was split out, and only room for it
    // when the original is being replaced outright (duck/dual already keep
    // the whole original, per the hint above). The box stays on screen and
    // tagged rather than hidden when overridden, so the override has a reason
    // attached instead of just silently doing nothing.
    //
    // Dimmed rather than disabled: a disabled control drops out of the tab
    // order, so a keyboard user could never reach this field, its tag, or its
    // hint at all.
    const separating = this.$("#separate_audio").value === "true";
    const overridden = separating && !replacing;
    this.$("#keepMusicBox").classList.toggle("hidden", !separating);
    this.$("#keep_music").classList.toggle("dim", overridden);
    this.$("#keepMusicTag").classList.toggle("hidden", !overridden);
    const hint = this.$("#keepMusicHint");
    hint.classList.toggle("hidden", !overridden);
    hint.textContent = overridden
      ? "There's no room to mix a separate copy of the music in while the "
        + "whole original is already staying in the file. Set Original audio "
        + "to Replace completely for the music and effects under the new "
        + "voices without the original speech."
      : "";
  }

  save(){
    const data = {};
    TEXT_SETTINGS.forEach(k => { if(this.$(`#${k}`)) data[k] = this.$(`#${k}`).value; });
    BOOL_SETTINGS.forEach(k => { if(this.$(`#${k}`)) data[k] = this.$(`#${k}`).value === "true"; });
    NUM_SETTINGS.forEach(k => { if(this.$(`#${k}`)) data[k] = parseFloat(this.$(`#${k}`).value); });
    Object.assign(data, this.audioChoice());
    if(this._language) data.target_language = this._language;
    this.emit("save-settings", {data});
  }

  audioChoice(){
    const [mode, level] = this.$("#audio_mode").value.split(":");
    const data = {audio_mode: mode};
    // Only duck mode reads a level, so any other mode puts it back to the
    // shipped one rather than leaving a level saved that nothing consults and
    // no control shows.
    const db = mode === "duck" ? parseFloat(level) : this._defaults?.duck_db;
    if(db !== undefined) data.duck_db = db;
    return data;
  }

  resetTab(btn){
    const tab = this.tab();
    const seen = this.visibleFields(tab).length;
    const lines = [
      `Reset the ${seen} setting${seen === 1 ? "" : "s"} on ${tab.label} to what the `
      + `app ships with?`,
      "",
      "It happens straight away — there is nothing to save afterwards. Nothing on the "
      + "other tabs changes, and finished videos are untouched.",
    ];
    if(tab.key === "translation"){
      lines.push("", "Your saved API keys are left alone. To remove one, empty the box "
        + "and press Save.");
      // Named in the request, so the server's own guard does not hold it back —
      // and it is the one field on this tab that was typed out by hand and
      // cannot be chosen again from a menu.
      if((this._settings?.custom_glossary || "").trim()){
        lines.push("", "Your own terms are on this tab, and they will be cleared. "
          + "There is no undo — copy them somewhere first if you want to keep them.");
      }
    }
    this.ask(btn, lines, this.fields(tab));
  }

  resetAll(btn){
    this.ask(btn, [
      "Reset every setting to what the app ships with?",
      "",
      "That includes the ones outside this window — who's speaking, and keeping the "
      + "Mac awake. It happens straight away.",
      "",
      "Your saved API keys and your finished videos are left alone.",
    ], null);
  }

  // A reset is server-side and immediate, while Save submits the whole form, so
  // the two would fight over a field the user has edited but not saved: the
  // fields being reset are repainted from the reply, and any edit outside that
  // list stays on screen for Save to send.
  ask(btn, lines, keys){
    const scope = keys || TABS.flatMap(t => this.fields(t));
    const losing = scope.filter(k => this._dirty.has(k) && this.shown(k)).length;
    if(losing){
      lines.push("", `${losing} change${losing === 1 ? "" : "s"} you haven't saved yet `
        + `${losing === 1 ? "is" : "are"} among them, and will be overwritten.`);
    }
    if(!confirm(lines.join("\n"))) return;
    scope.forEach(k => this._dirty.delete(k));
    // Both reset buttons answer to the same busy group, so the one that was
    // pressed travels with the request rather than the group that would label
    // them both.
    this.emit("reset-settings", {keys, btn});
  }

  showDiagnostics(){
    // Two modal dialogs stack, and the one underneath goes inert and stays
    // dimmed behind the new backdrop; closing the details returns here.
    document.querySelector("diagnostics-panel")?.open();
  }

  showSaved(){
    // The reply arrives while the edits that produced it are still marked
    // unsaved, so the fields that carried them kept what was typed rather than
    // what was stored. Now that nothing is unsaved they can be painted from what
    // the server actually kept.
    const unsaved = this._dirty.size;
    this._dirty.clear();
    if(unsaved && this._state) this.repaint(this._state);
    this.note("Saved");
  }

  showReset(){ this.note("Back to defaults"); }

  // Left up until something replaces it: a save that did not happen is not news
  // to be missed while looking elsewhere.
  showSaveError(message){
    this.note(message, 0, true);
  }

  note(text, ms, bad){
    const el = this.$("#savedMsg");
    el.classList.toggle("bad", !!bad);
    flash(el, text, ms);
  }

  async audition(){
    const btn = this.$("#auditionBtn"), msg = this.$("#auditionMsg"), el = this.$("#auditionAudio");
    const voice = this.$("#voice").value, speed = this.$("#speed").value;
    btn.disabled = true;
    // The first click loads the speech model, which takes a few seconds; say so
    // rather than looking like the button did nothing.
    msg.textContent = "Speaking…";
    try{
      const blob = await api.voicePreview(voice, speed);
      el.src = URL.createObjectURL(blob);
      await el.play();
      msg.textContent = "";
    }catch(err){
      msg.textContent = err.message;
    }finally{
      btn.disabled = false;
    }
  }
}

customElements.define("settings-panel", SettingsPanel);
