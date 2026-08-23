import { BaseElement } from "../base-element.js";
import { store } from "../store.js";
import { niceDate } from "../format.js";
import "./info-tip.js";

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
      <button class="toggle hidden" id="aAwake"></button><info-tip
        id="awakeTip" class="hidden" label="staying awake"></info-tip>
      <button class="icon-btn" id="settingsBtn">Settings</button>
    </div>
  </header>
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
      if(!confirm("Update Dubbing Studio?\n\nThe installer opens in Terminal. "
                + "Settings and finished videos are kept."))
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
    // macOS only: caffeinate is what holds the machine awake, and it is Apple's.
    const here = s.machine?.system === "Darwin";
    this.$("#aAwake").classList.toggle("hidden", !here);
    this.$("#awakeTip").classList.toggle("hidden", !here);
    this.renderAwake(!!s.settings.keep_awake);
  }

  // The pill alone read as a status badge rather than a control — a tester
  // clicked it just to find out what it was, and silently gave up the thing
  // stopping her Mac napping through a long dub. The info-tip beside it says
  // so before that click, the same way the tips inside Settings explain a
  // setting before it's changed.
  renderAwake(on){
    const el = this.$("#aAwake");
    el.setAttribute("aria-pressed", String(on));
    el.innerHTML = MOON.replace("%SLASH%", on ? SLASH : "")
      + (on ? "Won't sleep" : "May sleep");
    this.$("#awakeTip").text = on
      ? "Your Mac is kept awake while a video is dubbing. The screen can still "
        + "sleep, and closing the lid still sleeps. Click to allow sleep."
      : "Your Mac may sleep while a video is dubbing, which pauses the job "
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
