import { BaseElement } from "../base-element.js";

// The popover lives in the top layer, which is the only place a panel can be
// drawn from inside a scrolling dialog body without being clipped by it. The
// top layer costs it its anchor: its containing block is the viewport, so the
// position has to be measured and written by hand each time it opens.
const SHELL = `
<style>
  :host{display:inline-block;vertical-align:middle;margin-left:4px}
  .tip-btn{
    width:16px;height:16px;padding:0;
    display:grid;place-items:center;
    font-family:ui-serif,Georgia,"Times New Roman",serif;
    font-size:11px;font-style:italic;font-weight:700;line-height:1;
    border:1px solid var(--line);border-radius:99px;
    background:transparent;color:var(--muted)}
  .tip-btn:hover{color:var(--accent);border-color:var(--accent)}
  .tip-btn[aria-expanded="true"]{color:var(--accent);border-color:var(--accent);
                                 background:var(--accent-soft)}
  .tip{
    position:fixed;inset:auto;margin:0;
    width:max-content;max-width:min(300px,calc(100vw - 24px));
    max-height:min(50dvh,320px);
    overflow-y:auto;overscroll-behavior:contain;
    padding:10px 12px;
    /* Normalised rather than inherited: the tip usually sits inside a <label>,
       whose 600 weight, muted colour and letter-spacing reach across the
       shadow boundary. */
    font-size:12.5px;line-height:1.45;font-weight:400;font-style:normal;
    letter-spacing:normal;text-align:left;white-space:pre-line;
    color:var(--ink);background:var(--panel);
    border:1px solid var(--line);border-radius:var(--radius);box-shadow:var(--shadow)}
  /* Shown before it is measured, so the first frame would otherwise land at
     the top-layer default position rather than beside the button. */
  .tip:not(.placed){opacity:0}
</style>
<button class="tip-btn" type="button" aria-expanded="false">i</button>
<div class="tip" popover role="note" tabindex="0"></div>
`;

let seq = 0;

class InfoTip extends BaseElement {
  connectedCallback(){
    if(!this._built){
      this._built = true;
      this.html(SHELL);
      const id = `tip-${++seq}`;
      const btn = this.$(".tip-btn"), tip = this.$(".tip");
      tip.id = id;
      btn.setAttribute("popovertarget", id);
      btn.setAttribute("aria-controls", id);

      // A label with a `for` treats a click anywhere inside it as a click on
      // the field, and the button is a shadow child, so the retargeted event
      // does not read as the interactive descendant that would exempt it.
      btn.addEventListener("click", e => e.stopPropagation());
      tip.addEventListener("beforetoggle", e => this.toggled(e.newState === "open"));
      tip.addEventListener("toggle", e => { if(e.newState === "open") this.place(); });

      this.paint();
    }
  }

  disconnectedCallback(){
    this.track(false);
  }

  get text(){ return this.getAttribute("text") || ""; }
  set text(v){
    this.setAttribute("text", v ?? "");
    if(this._built) this.paint();
  }

  get label(){ return this.getAttribute("label") || ""; }

  get open(){ return this.$(".tip").matches(":popover-open"); }

  close(){ if(this.open) this.$(".tip").hidePopover(); }

  paint(){
    this.$(".tip").textContent = this.text;
    this.$(".tip-btn").setAttribute("aria-label",
      this.label ? `More about ${this.label}` : "More information");
  }

  toggled(isOpen){
    this.$(".tip-btn").setAttribute("aria-expanded", String(isOpen));
    if(!isOpen){
      const tip = this.$(".tip");
      tip.classList.remove("placed");
      // A popover hands focus back to whatever had it before, but a scroll
      // container inside one is left to hand it back itself: closed while the
      // tip has focus, the tab position drops to the body.
      if(this.shadowRoot.activeElement === tip) this.$(".tip-btn").focus();
    }
    this.track(isOpen);
  }

  // A scroll event stays inside the tree it happened in, and the scroller that
  // carries the button is the dialog body a tree further out — so one listener
  // per tree on the way out, rather than one on the document, which never hears
  // it. The tip's own tree is not among them, which is what leaves a long tip
  // free to scroll inside itself.
  track(on){
    this._tracking?.abort();
    if(!on) return;
    const { signal } = (this._tracking = new AbortController());
    const move = () => { if(this.open) this.place(); };
    for(let root = this.getRootNode(); root; root = root.host?.getRootNode()){
      root.addEventListener("scroll", () => this.close(),
                            {capture: true, passive: true, signal});
    }
    addEventListener("resize", move, {signal});
    // A hidden ancestor — a tab pane, a collapsed form — takes the button's box
    // away without moving anything, so no scroll reports it.
    const boxed = new ResizeObserver(move);
    boxed.observe(this.$(".tip-btn"));
    signal.addEventListener("abort", () => boxed.disconnect(), {once: true});
  }

  place(){
    const tip = this.$(".tip");
    const r = this.$(".tip-btn").getBoundingClientRect();
    // Out of the window, or left with no box at all by a hidden ancestor: the
    // tip has nothing left to point at, and where it stands it reads as
    // belonging to whatever is now beneath it.
    if(!r.width || !r.height || r.bottom < 0 || r.top > innerHeight){ this.close(); return; }

    const gap = 8, edge = 8;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let top = r.bottom + gap;
    if(top + h > innerHeight - edge && r.top - gap - h > edge) top = r.top - gap - h;
    top = Math.max(edge, Math.min(top, innerHeight - edge - h));
    const left = Math.max(edge,
      Math.min(r.left + r.width / 2 - w / 2, innerWidth - edge - w));

    tip.style.top = `${Math.round(top)}px`;
    tip.style.left = `${Math.round(left)}px`;
    tip.classList.add("placed");
  }
}

customElements.define("info-tip", InfoTip);
