import { BaseElement } from "../base-element.js";

const SHELL = `
  <div class="past">
    <div style="min-width:0;display:flex;align-items:flex-start">
      <span class="swatch" id="swatch" style="margin-top:5px;display:none"></span>
      <div style="min-width:0">
        <b><span id="title"></span><span class="qty" id="qty" style="display:none"></span><span class="tag" id="badge" style="display:none"></span></b>
        <small id="subtitle"></small>
      </div>
    </div>
    <div id="actions" style="display:flex;gap:8px;flex-shrink:0"></div>
  </div>
`;

class ListRow extends BaseElement {
  connectedCallback(){
    this.html(SHELL);
    if(this._data) this._render();
  }

  set data(value){
    this._data = value;
    if(this.shadowRoot.childElementCount) this._render();
  }

  get data(){
    return this._data;
  }

  _render(){
    const d = this._data || {};

    const swatch = this.$("#swatch");
    if(d.color){ swatch.style.background = d.color; swatch.style.display = "inline-block"; }
    else swatch.style.display = "none";

    this.$("#title").textContent = d.title || "";

    const qty = this.$("#qty");
    if(d.qty){ qty.textContent = d.qty; qty.style.display = "inline"; }
    else qty.style.display = "none";

    const badge = this.$("#badge");
    if(d.badge){ badge.textContent = d.badge; badge.style.display = "inline-block"; }
    else badge.style.display = "none";

    this.$("#subtitle").textContent = d.subtitle || "";

    const actions = this.$("#actions");
    actions.innerHTML = "";
    (d.actions || []).forEach(a => {
      const btn = document.createElement("button");
      btn.className = a.className || "ghost";
      btn.textContent = a.label;
      btn.disabled = !!a.disabled;
      btn.onclick = () => a.onClick(btn);
      actions.appendChild(btn);
    });
  }
}

customElements.define("list-row", ListRow);
