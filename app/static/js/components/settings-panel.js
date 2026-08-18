import { BaseElement } from "../base-element.js";
import { store } from "../store.js";
import { api } from "../api.js";

const BOOL_SETTINGS = ["write_srt","separate_audio","diarize","keep_music","merge_lines"];
const NUM_SETTINGS = {speed: 1.0, expected_speakers: -1, duck_db: -18, max_stretch: 1.55};
const TEXT_SETTINGS = ["voice","audio_mode","glossary","translator","ollama_model",
  "youtube_cookies",
  "anthropic_key","openai_key","anthropic_model","openai_model","target_language",
  "custom_glossary","keep_video_quality","asr_model","voice_mode"];

const SHELL = `
<dialog id="dlg">
  <div class="panel" style="margin:0;max-width:640px;max-height:80vh;overflow:auto">
    <div class="grid">
      <div>
        <label for="voice">Voice</label>
        <select id="voice"></select>
        <div style="display:flex;gap:8px;align-items:center;margin-top:8px">
          <button class="ghost" id="auditionBtn">Hear this voice</button>
          <span class="hint" id="auditionMsg" style="margin:0" role="status" aria-live="polite"></span>
        </div>
        <audio id="auditionAudio" style="display:none"></audio>
      </div>
      <div>
        <label for="audio_mode">Original audio</label>
        <select id="audio_mode">
          <option value="replace">Replace completely</option>
          <option value="duck">Keep quietly underneath</option>
          <option value="dual">Keep as a second track</option>
        </select>
      </div>
      <div>
        <label for="translator">Translated by</label>
        <select id="translator">
          <option value="ollama">Local model (free, private)</option>
          <option value="anthropic">Claude API (best quality)</option>
          <option value="openai">OpenAI API</option>
        </select>
      </div>
      <div id="ollamaBox">
        <label for="ollama_model">Local model</label>
        <input type="text" id="ollama_model" placeholder="auto">
        <p class="hint" id="modelHint"></p>
      </div>
      <div id="anthropicBox" class="hidden">
        <label for="anthropic_key">Anthropic API key</label>
        <input type="password" id="anthropic_key" placeholder="sk-ant-…">
        <label for="anthropic_model" style="margin-top:10px">Claude model</label>
        <input type="text" id="anthropic_model" placeholder="claude-sonnet-5">
        <p class="hint">The key is saved in plain text in
          <code id="settingsPath1">settings.json</code> on this computer, so that
          the app can translate without asking for it again. Anyone with access to
          your account can read it.</p>
      </div>
      <div id="openaiBox" class="hidden">
        <label for="openai_key">OpenAI API key</label>
        <input type="password" id="openai_key" placeholder="sk-…">
        <label for="openai_model" style="margin-top:10px">OpenAI model</label>
        <input type="text" id="openai_model" placeholder="gpt-4o">
        <p class="hint">The key is saved in plain text in
          <code id="settingsPath2">settings.json</code> on this computer, so that
          the app can translate without asking for it again. Anyone with access to
          your account can read it.</p>
      </div>
      <div>
        <label for="target_language">Translate into</label>
        <input type="text" id="target_language" value="English">
      </div>
      <div>
        <label for="speed">Speaking speed</label>
        <select id="speed">
          <option value="0.9">Slower</option>
          <option value="1.0">Normal</option>
          <option value="1.1">Slightly faster</option>
        </select>
      </div>
    </div>

    <details style="margin-top:18px">
      <summary>Advanced</summary>
      <div style="margin-top:12px">
        <label for="custom_glossary">Your own terms (one per line, “as spoken → English”)</label>
        <textarea id="custom_glossary" placeholder="ponto amêndoa -> almond stitch"></textarea>
        <div class="grid" style="margin-top:14px">
          <div>
            <label for="separate_audio">Music and effects</label>
            <select id="separate_audio">
              <option value="true">Separate and keep them</option>
              <option value="false">Don't separate (faster)</option>
            </select>
            <p class="hint">Splits speech from the rest so replacing the voices
              doesn't wipe the soundtrack.</p>
          </div>
          <div>
            <label for="expected_speakers">How many people speak</label>
            <select id="expected_speakers">
              <option value="-1">Work it out automatically</option>
              <option value="1">1 person</option>
              <option value="2">2 people</option>
              <option value="3">3 people</option>
              <option value="4">4 people</option>
              <option value="5">5 people</option>
              <option value="6">6 people</option>
            </select>
            <p class="hint">Only used when “Who's speaking?” is set to several
              people. Working it out automatically is the least reliable part of
              the chain, so if you know the answer, saying so here is better.</p>
          </div>
          <div>
            <label for="keep_music">Music and effects bed</label>
            <select id="keep_music">
              <option value="true">Mix it back under the new voices</option>
              <option value="false">Drop it — voices only</option>
            </select>
            <p class="hint">Only applies when the soundtrack was separated and the
              original audio is being replaced.</p>
          </div>
          <div>
            <label for="asr_model">Transcription engine</label>
            <select id="asr_model">
              <option value="parakeet">Parakeet — fast</option>
              <option value="whisper">Whisper — more accurate, slower</option>
            </select>
          </div>
          <div>
            <label for="voice_mode">Voices</label>
            <select id="voice_mode">
              <option value="fixed">Use a built-in voice</option>
              <option value="clone">Clone the original speaker</option>
            </select>
            <p class="hint" id="cloneHint">Cloning keeps the speaker's identity, but
              carries their accent into English.</p>
          </div>
        </div>
        <div class="grid" style="margin-top:14px">
          <div>
            <label for="youtube_cookies">Sign in as</label>
            <select id="youtube_cookies">
              <option value="">Don't sign in</option>
              <option value="safari">Safari</option>
              <option value="chrome">Chrome</option>
              <option value="firefox">Firefox</option>
              <option value="edge">Edge</option>
              <option value="brave">Brave</option>
            </select>
            <p class="hint">For videos YouTube refuses to send to a signed-out
              request. Borrows the session from a browser you're already signed
              into on this Mac. Nothing is uploaded; the cookies are only sent to
              the site the video is on.</p>
          </div>
          <div>
            <label for="keep_video_quality">Video quality</label>
            <select id="keep_video_quality">
              <option value="best">Best available</option>
              <option value="1080">Up to 1080p</option>
              <option value="720">Up to 720p — much smaller</option>
            </select>
            <p class="hint">The picture is copied, never re-encoded, so this
              decides the download size. On an hour-and-a-half tutorial, 1080p is
              around 1.8 GB and 720p around 730 MB — and 720p is plenty for
              following along.</p>
          </div>
          <div>
            <label for="write_srt">Subtitles</label>
            <select id="write_srt">
              <option value="false">No subtitle file</option>
              <option value="true">Also save an .srt</option>
            </select>
          </div>
          <div>
            <label for="duck_db">Original volume when kept underneath</label>
            <select id="duck_db">
              <option value="-12">Quiet (-12 dB)</option>
              <option value="-18">Very quiet (-18 dB)</option>
              <option value="-24">Barely there (-24 dB)</option>
            </select>
            <p class="hint">Used only by “Keep quietly underneath”.</p>
          </div>
          <div>
            <label for="merge_lines">Run-on lines</label>
            <select id="merge_lines">
              <option value="true">Join lines that run together</option>
              <option value="false">Keep them exactly as heard</option>
            </select>
            <p class="hint">Fast dialogue arrives as many very short lines with no
              gap between them, and each has to be squeezed to fit. Joining them
              gives the translation room. Material with real pauses is unaffected.</p>
          </div>
          <div>
            <label for="max_stretch">Hardest allowed squeeze</label>
            <select id="max_stretch">
              <option value="1.3">Gentle — 1.3x</option>
              <option value="1.55">Normal — 1.55x</option>
              <option value="1.8">Firm — 1.8x</option>
            </select>
            <p class="hint">How much a line may be sped up to fit the gap the
              original speaker left. Past about 1.6x it starts to sound hurried;
              beyond the limit the line runs on and later pauses absorb it.</p>
          </div>
        </div>
      </div>
    </details>

    <div style="margin-top:20px;display:flex;gap:8px">
      <button class="primary" id="saveBtn" data-busy="save">Save</button>
      <button class="ghost" id="closeBtn">Close</button>
      <span id="savedMsg" class="hint" style="align-self:center;margin:0" role="status" aria-live="polite"></span>
    </div>
  </div>
</dialog>
`;

class SettingsPanel extends BaseElement {
  connectedCallback(){
    this.html(`<style>
      dialog{border:none;border-radius:var(--radius);padding:0;background:transparent;
             max-width:640px;width:calc(100vw - 48px)}
      dialog::backdrop{background:rgba(0,0,0,.4)}
    </style>` + SHELL);

    this.$("#translator").addEventListener("change", () => this.translatorChanged());
    this.$("#saveBtn").onclick = () => this.save();
    this.$("#closeBtn").onclick = () => this.close();
    this.$("#auditionBtn").onclick = () => this.audition();

    this._unsub = store.subscribe(s => this.update(s));
  }

  disconnectedCallback(){
    this._unsub?.();
  }

  open(){ this.$("#dlg").showModal(); }
  close(){ this.$("#dlg").close(); }

  update(s){
    const { settings, voices, machine, settings_path } = s;
    if(!Object.keys(settings).length) return;

    this.$("#voice").innerHTML = (voices || [])
      .map(o=>`<option value="${o.id}">${o.label}</option>`).join("");

    TEXT_SETTINGS.forEach(k => { if(this.$(`#${k}`)) this.$(`#${k}`).value = settings[k] ?? ""; });
    BOOL_SETTINGS.forEach(k => { if(this.$(`#${k}`)) this.$(`#${k}`).value = String(!!settings[k]); });
    // Matched numerically, not as a string. JSON turns Python's 1.0 into JS 1,
    // and String(1) is "1", which matches none of the "0.9"/"1.0"/"1.1" options.
    Object.entries(NUM_SETTINGS).forEach(([k, dflt]) => {
      const el = this.$(`#${k}`);
      if(!el) return;
      const want = Number(settings[k] ?? dflt);
      const opt = el.tagName === "SELECT"
        ? [...el.options].find(o => Number(o.value) === want) : null;
      el.value = opt ? opt.value : (el.tagName === "SELECT" ? el.value : String(want));
    });

    this.translatorChanged();

    this.$("#settingsPath1").textContent = settings_path || "settings.json";
    this.$("#settingsPath2").textContent = settings_path || "settings.json";

    if(machine && machine.suggested_model){
      this.$("#modelHint").textContent =
        `Leave blank to use ${machine.suggested_model}, chosen for your `
        + `${machine.ram_gb} GB of memory. You can name any model Ollama has — `
        + `qwen3:32b translates specialist material better, but it is a 20 GB `
        + `download and slower per line. Pull it in Ollama first; if it isn't there, `
        + `the nearest installed model is used instead and the report says so.`;
    }
  }

  translatorChanged(){
    const t = this.$("#translator").value;
    this.$("#ollamaBox").classList.toggle("hidden", t!=="ollama");
    this.$("#anthropicBox").classList.toggle("hidden", t!=="anthropic");
    this.$("#openaiBox").classList.toggle("hidden", t!=="openai");
  }

  save(){
    const data = {};
    TEXT_SETTINGS.forEach(k => { if(this.$(`#${k}`)) data[k] = this.$(`#${k}`).value; });
    BOOL_SETTINGS.forEach(k => { if(this.$(`#${k}`)) data[k] = this.$(`#${k}`).value === "true"; });
    Object.keys(NUM_SETTINGS).forEach(k => { if(this.$(`#${k}`)) data[k] = parseFloat(this.$(`#${k}`).value); });
    this.emit("save-settings", {data});
  }

  showSaved(){
    const msg = this.$("#savedMsg");
    msg.textContent = "Saved";
    setTimeout(()=>{ if(msg.textContent === "Saved") msg.textContent = ""; }, 1800);
  }

  showSaveError(message){
    this.$("#savedMsg").textContent = message;
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
