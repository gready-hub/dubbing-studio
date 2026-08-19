import { BaseElement } from "../base-element.js";
import { store } from "../store.js";
import "./queue-list.js";
import "./history-list.js";
import "./storage-panel.js";
import "./doctor-panel.js";

const TABS = [
  {key: "history", label: "History"},
  {key: "storage", label: "Storage"},
  {key: "doctor", label: "Setup check"},
];

const SHELL = `
  <queue-list></queue-list>
  <div class="segmented" id="tabs" style="margin-bottom:14px"></div>
  <history-list></history-list>
  <storage-panel></storage-panel>
  <doctor-panel></doctor-panel>
`;

class ManagePanel extends BaseElement {
  connectedCallback(){
    this.html(SHELL);
    this.$("#tabs").innerHTML = TABS.map(t =>
      `<button data-tab="${t.key}"><span class="mark" aria-hidden="true">✓</span>${
        t.label}</button>`).join("");
    this.$$("#tabs button").forEach(b => {
      b.onclick = () => this.selectTab(b.dataset.tab, true);
    });
    this._userPicked = false;
    this._unsub = store.subscribe(s => this.update(s));
  }

  disconnectedCallback(){
    this._unsub?.();
  }

  showStorageCheckError(){
    this.$("storage-panel").showCheckError();
  }

  // The banner that says a dependency is missing points at this tab by name.
  revealDoctor(){
    this.selectTab("doctor", false);
  }

  update(s){
    if(this._userPicked) return;
    // Keeps re-checking on every update, not just the first, since doctor and
    // storage data arrive asynchronously after boot — a tab is only "picked"
    // once the user actually clicks one.
    const wanted = s.doctor && !s.doctor.ready ? "doctor"
      : s.storage && s.storage.low ? "storage"
      : "history";
    this.selectTab(wanted, false);
  }

  selectTab(key, manual){
    if(manual) this._userPicked = true;
    this.$$("#tabs button").forEach(b => {
      const on = b.dataset.tab === key;
      b.classList.toggle("on", on);
      b.setAttribute("aria-pressed", String(on));
    });
    this.$("history-list").hidden = key !== "history";
    this.$("storage-panel").hidden = key !== "storage";
    this.$("doctor-panel").hidden = key !== "doctor";
  }
}

customElements.define("manage-panel", ManagePanel);
