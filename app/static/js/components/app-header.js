import { BaseElement } from "../base-element.js";
import { store } from "../store.js";
import { niceDate } from "../format.js";

// One shape either way: a crescent moon, struck through when sleep is being
// held off. It carries the meaning without colour, which the pill's border
// alone would not.
const MOON = '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor"'
  + ' stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">'
  + '<path d="M14 9.3A6.2 6.2 0 0 1 6.7 2 6 6 0 1 0 14 9.3z"/>%SLASH%</svg>';
const SLASH = '<path d="M2.2 13.8 13.8 2.2"/>';

const SHELL = `
  <header>
    <div><h1>Dubbing Studio</h1></div>
    <div>
      <button class="toggle hidden" id="aAwake"></button>
      <button class="icon-btn" id="settingsBtn">Settings</button>
    </div>
  </header>
  <p class="sub">Paste a video link. Get it back speaking English.</p>
  <div id="updateBanner" class="banner info hidden"
       style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">
    <span id="updateText" style="flex:1;min-width:180px"></span>
    <button class="ghost" style="padding:6px 12px;font-size:13px" id="updateNow">Update now</button>
    <button class="ghost" style="padding:6px 12px;font-size:13px" id="updateDismiss">Not now</button>
  </div>
`;

class AppHeader extends BaseElement {
  connectedCallback(){
    this.html(SHELL);
    this.$("#aAwake").onclick = () => this.toggleAwake();
    this.$("#settingsBtn").onclick = () => document.querySelector("settings-panel")?.open();
    this.$("#updateNow").onclick = () => {
      if(!confirm("Update Dubbing Studio?\n\nThis opens the installer in Terminal. "
                + "Your settings, videos and finished work are kept."))
        return;
      this.emit("run-update");
    };
    this.$("#updateDismiss").onclick = () => this.$("#updateBanner").classList.add("hidden");
    this._unsub = store.subscribe(s => this.update(s));
  }

  disconnectedCallback(){
    this._unsub?.();
  }

  update(s){
    // macOS only: caffeinate is its, and the Docker build holds nothing.
    this.$("#aAwake").classList.toggle("hidden", s.machine?.system !== "Darwin");
    this.renderAwake(!!s.settings.keep_awake);
  }

  renderAwake(on){
    const el = this.$("#aAwake");
    el.setAttribute("aria-pressed", String(on));
    el.innerHTML = MOON.replace("%SLASH%", on ? SLASH : "")
      + (on ? "Won't sleep" : "May sleep");
    el.title = on
      ? "While a video is being dubbed, your Mac is kept awake so the job isn't "
        + "left half-done. The screen can still sleep; closing the lid still "
        + "sleeps. Click to allow it to sleep."
      : "Your Mac may sleep part-way through dubbing a video, which pauses the job "
        + "until you wake it. Click to keep it awake.";
  }

  toggleAwake(){
    const on = !store.state.settings.keep_awake;
    this.renderAwake(on);
    this.emit("toggle-awake", {on});
  }

  showUpdate(v){
    this.$("#updateText").textContent = v.date
      ? `An update is available, from ${niceDate(v.date)}.`
      : "An update is available.";
    this.$("#updateBanner").title = `${v.current} → ${v.latest}`;
    this.$("#updateBanner").classList.remove("hidden");
  }
}

customElements.define("app-header", AppHeader);
