import { flash } from "./format.js";

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

  // A live region re-announces whatever is written into it, and writing the
  // same words again still counts as writing. Progress arrives about twice a
  // second, so the announcement has to be spared when nothing has changed.
  say(selector, text){
    const el = this.$(selector);
    if(el && el.textContent !== text) el.textContent = text;
  }

  renderIfChanged(sig, fn){
    const key = JSON.stringify(sig);
    if(this._sig === key) return false;
    this._sig = key;
    fn();
    return true;
  }

  // Marks whichever button in a group is the current choice — the shape a
  // segmented control or a tab strip always wants, whatever decides which one
  // that is.
  markOn(selector, test){
    this.$$(selector).forEach(b => {
      const on = test(b);
      b.classList.toggle("on", on);
      b.setAttribute("aria-pressed", String(on));
    });
  }

  // A status line that flags itself as bad and then fades on its own — the
  // shape "saved"/"saving"/"save failed" always take, whichever line it is.
  flashNote(selector, text, ms, bad){
    const el = this.$(selector);
    el.classList.toggle("bad", !!bad);
    flash(el, text, ms);
  }

  // `target` says which buttons this is about: an element or list of them for
  // the exact ones — the button that was pressed, say, out of several that
  // share a group — a string for every [data-busy] answering to that group, and
  // nothing for all of them.
  //
  // The label each button carries is remembered against the button itself, so
  // two overlapping calls that reach the same one restore what it said before
  // either of them rather than what the first left on it.
  setBusy(isBusy, label, target){
    const targets = target instanceof Element ? [target]
      : typeof target === "string" ? this.$$(`[data-busy="${target}"]`)
      : target ? [...target]
      : this.$$("[data-busy]");
    if(!targets.length) return;
    this._busyRestore ??= new WeakMap();
    targets.forEach(b => {
      b.disabled = isBusy;
      if(isBusy){
        if(!this._busyRestore.has(b)) this._busyRestore.set(b, b.textContent);
        if(label) b.textContent = label;
      } else if(this._busyRestore.has(b)){
        b.textContent = this._busyRestore.get(b);
        this._busyRestore.delete(b);
      }
    });
  }
}
