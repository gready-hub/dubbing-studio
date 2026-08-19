import { BaseElement } from "../base-element.js";

const HEAD = `
  <span class="caret" aria-hidden="true"></span>
  <span class="swatch" id="swatch" style="margin-top:5px;display:none"></span>
  <span class="lines">
    <b><span id="title"></span><span class="qty" id="qty" style="display:none"></span><span class="tag" id="badge" style="display:none"></span></b>
    <small id="subtitle"></small>
  </span>
`;

const SHELL = `
  <style>
    /* Drawn above rather than below, because a row cannot see whether it is the
       last of its list but :host(:first-child) tells it when it is the first.
       The shared .past:last-child rule cannot do it: inside the shadow root
       .past has the detail drawer after it and so is never last. */
    :host{display:block;border-top:1px solid var(--line)}
    :host(:first-child){border-top:none}
    .past{border-bottom:none}
    .head{flex:1;min-width:0;display:flex;align-items:flex-start}
    .lines{min-width:0;flex:1}
    .caret{display:none}
    button.head{font:inherit;font-size:14px;font-weight:inherit;color:inherit;
                text-align:left;background:none;border:none;padding:0;
                border-radius:8px;cursor:pointer}
    button.head .caret{display:block;flex:none;width:18px;color:var(--muted)}
    button.head .caret::before{content:"▸"}
    button.head[aria-expanded="true"] .caret::before{content:"▾"}
    button.head:hover .caret{color:var(--ink)}
    .detail{padding:2px 0 14px 18px;font-size:14px}
  </style>
  <div class="past">
    <div id="actions" style="display:flex;gap:8px;flex-shrink:0"></div>
  </div>
  <div class="detail" id="detail" hidden></div>
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

  // Public, because a list that keeps one row open at a time has to be able to
  // shut the others: a row knows nothing about the rows beside it.
  setOpen(open){
    const wanted = !!(open && this._data && this._data.detail);
    if(this._data) this._data.open = wanted;
    this.$("#detail").hidden = !wanted;
    this._toggle?.setAttribute("aria-expanded", String(wanted));
  }

  _render(){
    const d = this._data || {};
    if(!this.$("#head")) this._buildHead(!!d.detail);

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

    this.$("#detail").innerHTML = d.detail || "";
    this.setOpen(d.open);

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

  // A row with nothing to reveal must not be a button: it would take a tab stop
  // and announce itself as something to press.
  _buildHead(expandable){
    const tag = expandable ? "button" : "div";
    const attrs = expandable
      ? ' type="button" aria-controls="detail" aria-expanded="false"'
      : "";
    this.$("#actions").insertAdjacentHTML("beforebegin",
      `<${tag} class="head" id="head"${attrs}>${HEAD}</${tag}>`);
    this._toggle = expandable ? this.$("#head") : null;
    if(!this._toggle) return;
    this._toggle.onclick = () => {
      const open = !(this._data && this._data.open);
      this.setOpen(open);
      this._data?.onToggle?.(open);
    };
  }
}

customElements.define("list-row", ListRow);
