import { BaseElement } from "../base-element.js";
import { store } from "../store.js";
import { escapeHtml, escapeAttr, flash } from "../format.js";
import "./info-tip.js";

const PRESET_TIP =
  "A preset is a name for three settings: whether music and effects are "
  + "separated from the speech, which transcription engine runs, and whether "
  + "voices are built-in or cloned.\n\n"
  + "Everything else in Settings — voice, speed, translator, language — is "
  + "left as you set it.\n\n"
  + "Change any of the three and no preset stays highlighted; the line below "
  + "then says which one differs.";

const SPEAKERS_BLURB = {
  false: "One speaker can't be split into several voices — the usual mishap.",
  true: "Telling people apart is unreliable — check the result.",
};

const MARK = `<span class="mark" aria-hidden="true">✓</span>`;

const STYLE = `
<style>
  #speakersLabel{margin-top:18px}
  /* The label and its tip sit side by side rather than nested. Anything inside
     a <label> joins the accessible name of what the label names, and this one
     names the preset group — which would then announce the tip button too. */
  .label-row{display:flex;align-items:center;flex-wrap:wrap;margin-bottom:6px}
  .label-row label{margin-bottom:0}
  /* aria-disabled rather than disabled, so a blocked preset stays focusable:
     the reason it is blocked is the button's description, and a disabled
     button cannot be reached to hear it. */
  .segmented button[aria-disabled="true"]{opacity:.45;cursor:not-allowed}
  /* Its height is held whether or not it says anything, so acknowledging a save
     never moves the controls that were just used. */
  #saveMsg{min-height:1.45em;margin:10px 0 0}
  #saveMsg.bad{color:var(--bad)}
  #urlMsg{min-height:1.45em;margin:8px 0 0;color:var(--bad)}
</style>`;

const SHELL = `
  <div class="panel hidden" id="compactBar"
       style="display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap">
    <span class="msg" id="compactMsg"></span>
    <button class="ghost" id="expandBtn">Dub another video</button>
  </div>

  <div class="panel" id="fullForm">
    <label for="url">Video link</label>
    <div class="row">
      <input type="text" id="url" placeholder="https://www.youtube.com/watch?v=…"
             autocomplete="off" spellcheck="false">
      <button class="ghost" id="tryBtn" data-busy="start">Try 30 seconds</button>
      <button class="primary" id="go" data-busy="start">Dub it</button>
    </div>
    <p class="hint" id="urlMsg" role="status" aria-live="polite"></p>
    <p class="hint"><b>Try 30 seconds</b> dubs a short sample first, so you
      can hear the voice before waiting for a whole video.</p>

    <div class="preset-wrap">
      <div class="label-row">
        <label id="presetLabel">Quality</label>
        <info-tip id="presetTip" label="quality presets"
                  text="${escapeAttr(PRESET_TIP)}"></info-tip>
      </div>
      <div class="segmented" id="presets" role="group" aria-labelledby="presetLabel"></div>
      <p class="hint" id="presetBlurb"></p>
      <p class="hint hidden" id="presetBlocked" role="status" aria-live="polite"></p>

      <label id="speakersLabel">Who's speaking?</label>
      <div class="segmented" id="speakers" role="group" aria-labelledby="speakersLabel">
        <button data-diarize="false">${MARK}One person — faster, safer</button>
        <button data-diarize="true">${MARK}Several people — a voice each</button>
      </div>
      <p class="hint" id="speakersBlurb"></p>

      <p class="hint" id="saveMsg" role="status" aria-live="polite"></p>
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

// The three settings a preset is a name for, matching Settings.PRESET_KEYS on
// the server. Everything else in Settings is independent of the preset.
const PRESET_KEYS = ["separate_audio", "asr_model", "voice_mode"];

const DIFFERENCES = {
  separate_audio: {
    true: "music and effects separated from the speech",
    false: "music and effects left un-separated",
  },
  asr_model: {
    whisper: "Whisper transcribing instead of Parakeet",
    parakeet: "Parakeet transcribing instead of Whisper",
  },
  voice_mode: {
    clone: "each speaker's own voice cloned",
    fixed: "a built-in voice instead of cloning",
  },
};

const PRESET_SUMMARY =
  "Your own mix of the three settings a preset sets: music separation, the "
  + "transcription engine, and built-in versus cloned voices. Pick one above "
  + "to take all three from it.";

function differingKeys(spec, settings){
  return PRESET_KEYS.filter(k => spec[k] !== settings[k]);
}

// "custom" is a real state now that the preset is read back off the switches
// rather than left as whatever was last named. Nothing records which preset was
// departed from, so it is the nearest that gets named — which, for the presets
// that exist, is always a single switch away.
function customBlurb(presets, settings){
  const nearest = Object.values(presets || {})
    .map(p => ({p, keys: differingKeys(p, settings)}))
    .sort((a, b) => a.keys.length - b.keys.length)[0];
  if(!nearest) return PRESET_SUMMARY;
  if(!nearest.keys.length) return nearest.p.blurb;
  const parts = nearest.keys.map(k => DIFFERENCES[k]?.[String(settings[k])]);
  if(parts.some(part => !part)) return PRESET_SUMMARY;
  const list = parts.length > 1
    ? `${parts.slice(0, -1).join(", ")} and ${parts.at(-1)}`
    : parts[0];
  return `Your own mix: ${nearest.p.label}, but with ${list}. Pick a preset `
    + "above to go back to it, or change it in Settings.";
}

class NewJobPanel extends BaseElement {
  connectedCallback(){
    this.html(STYLE + SHELL);
    this.$("#tryBtn").onclick = () => this.start(true);
    this.$("#go").onclick = () => this.start(false);
    this.$("#url").addEventListener("keydown", e => { if(e.key === "Enter") this.start(false); });
    this.$("#url").addEventListener("input", () => flash(this.$("#urlMsg"), "", 0));
    this.$$("#speakers button").forEach(b => {
      b.onclick = () =>
        this.emit("save-settings", {data: {diarize: b.dataset.diarize === "true"}});
    });
    this.$("#expandBtn").onclick = () => {
      this._expanded = true;
      this.applyCollapse(store.state);
      this.$("#url").focus();
    };
    this._expanded = false;
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
    // A dead button is worse than a refusal: say what is missing rather than
    // letting the press do nothing at all.
    if(!url){
      flash(this.$("#urlMsg"), "Paste a video link first.", 5000);
      this.$("#url").focus();
      return;
    }
    flash(this.$("#urlMsg"), "", 0);
    this._expanded = false;
    this.emit("start-job", {url, preview: !!preview});
  }

  // The full form stays out of the way once a job exists — running, just
  // finished, or failed — so it isn't stacked on screen against the panel
  // that's actually relevant right now. "Dub another video" gets it back.
  applyCollapse(s){
    const job = s.jobs[s.current];
    // A cancelled job draws no panel of its own, so collapsing the form over it
    // leaves the window with nothing in it but the step chips. The link is still
    // in the box, and starting it again picks the work up where it stopped.
    const collapsed = !!job && job.status !== "cancelled" && !this._expanded;
    const form = this.$("#fullForm"), bar = this.$("#compactBar");
    // Whichever of the two is about to be taken off screen: focus left on a
    // hidden control drops to the body, and the swap goes both ways — a
    // cancelled job puts the form back with the focus still on "Dub another".
    const held = (collapsed ? form : bar).contains(this.shadowRoot.activeElement);
    form.classList.toggle("hidden", collapsed);
    bar.classList.toggle("hidden", !collapsed);
    if(collapsed){
      this.$("#compactMsg").textContent = job.status === "done" ? "That one's finished."
        : job.status === "error" ? "That one didn't finish."
        : "A video is dubbing.";
      if(held) this.$("#expandBtn").focus();
    } else if(held){
      this.$("#url").focus();
    }
  }

  update(s){
    this.applyCollapse(s);
    const { settings, presets, features } = s;
    this.renderIfChanged(
      [settings.diarize, features, PRESET_KEYS.map(k => settings[k])],
      () => {
        this.paintPresets(presets, settings, blockedReasons(features));
        this.paintSpeakers(!!settings.diarize);
      }
    );
  }

  // The buttons are built once and then only re-labelled, because rebuilding
  // them takes the focus off the one that was just activated.
  paintPresets(presets, settings, blocked){
    const keys = Object.keys(presets || {});
    if(keys.join(",") !== this._presetKeys){
      this._presetKeys = keys.join(",");
      const had = this.shadowRoot.activeElement?.dataset?.preset;
      this.$("#presets").innerHTML = keys.map(k =>
        `<button data-preset="${escapeAttr(k)}">${MARK}${escapeHtml(presets[k].label)}</button>`
      ).join("");
      this.$$("#presets button").forEach(b => {
        b.onclick = () => {
          if(b.getAttribute("aria-disabled") === "true") return;
          this.emit("save-settings", {data: {preset: b.dataset.preset}});
        };
      });
      if(had) this.$(`#presets button[data-preset="${CSS.escape(had)}"]`)?.focus();
    }

    const notes = Object.entries(blocked).filter(([, why]) => why);
    this.$("#presetBlocked").innerHTML = notes.map(([k, why]) =>
      `<span id="blocked-${escapeAttr(k)}">${escapeHtml(presets[k]?.label || k)}: `
      + `${escapeHtml(why)}</span>`).join(". ");
    this.$("#presetBlocked").classList.toggle("hidden", !notes.length);

    this.markOn("#presets button", b => b.dataset.preset === String(settings.preset));
    this.$$("#presets button").forEach(b => {
      const k = b.dataset.preset;
      if(blocked[k]){
        b.setAttribute("aria-disabled", "true");
        b.setAttribute("aria-describedby", `blocked-${k}`);
      } else {
        b.removeAttribute("aria-disabled");
        b.removeAttribute("aria-describedby");
      }
    });

    const p = (presets || {})[settings.preset];
    this.$("#presetBlurb").textContent = blocked[settings.preset]
      || (p ? p.blurb : customBlurb(presets, settings));
  }

  paintSpeakers(diarize){
    this.markOn("#speakers button", b => b.dataset.diarize === String(diarize));
    this.$("#speakersBlurb").textContent = SPEAKERS_BLURB[String(diarize)];
  }

  // These controls write straight through — there is no Save button here — so
  // the only thing that can say a change landed is the panel itself.
  showSaving(){
    this.flashNote("#saveMsg", "Saving…", 0);
  }

  showSaved(){
    this.flashNote("#saveMsg", "Saved");
  }

  // Left up until something replaces it: a save that did not happen is not news
  // to be missed while looking elsewhere.
  showSaveError(message){
    this.flashNote("#saveMsg", `Not saved — ${message}`, 0, true);
  }
}

customElements.define("new-job-panel", NewJobPanel);
