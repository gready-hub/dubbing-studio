const SHARED_STYLESHEET = `<link rel="stylesheet" href="/static/css/shared.css">`;

export class BaseElement extends HTMLElement {
  constructor(){
    super();
    this.attachShadow({mode: "open"});
  }

  $(selector){
    return this.shadowRoot.querySelector(selector);
  }

  $$(selector){
    return [...this.shadowRoot.querySelectorAll(selector)];
  }

  emit(name, detail){
    this.dispatchEvent(new CustomEvent(name, {detail, bubbles: true, composed: true}));
  }

  html(markup){
    this.shadowRoot.innerHTML = SHARED_STYLESHEET + markup;
  }

  renderIfChanged(sig, fn){
    const key = JSON.stringify(sig);
    if(this._sig === key) return false;
    this._sig = key;
    fn();
    return true;
  }

  setBusy(isBusy, label, group){
    const selector = group ? `[data-busy="${group}"]` : "[data-busy]";
    const targets = this.$$(selector);
    if(!targets.length) return;
    this._busyRestore ??= {};
    if(isBusy){
      this._busyRestore[group || ""] = targets.map(b => b.textContent);
      targets.forEach(b => {
        b.disabled = true;
        if(label) b.textContent = label;
      });
    } else {
      const restore = this._busyRestore[group || ""];
      targets.forEach((b, i) => {
        b.disabled = false;
        if(restore) b.textContent = restore[i];
      });
      delete this._busyRestore[group || ""];
    }
  }
}
