import { BaseElement } from "../base-element.js";
import "./list-row.js";

// The list-level concerns a capped, single-open disclosure list needs no
// matter what it lists: stable-identity diffing so an open row survives a
// repaint, the cap plus "…and N more", and closing whichever row was open
// when another one opens. list-row.js owns everything about one row; this is
// the seam beside it for everything about the list they sit in.
export class CappedList extends BaseElement {
  connectedCallback(){
    this._rows = new Map();
  }

  disconnectedCallback(){
    this._unsub?.();
  }

  // `items` is already filtered and sorted; `toRow(item)` returns
  // {id, sig, data} for list-row, leaving out `open`/`onToggle` since those
  // are this class's job. `moreText(rest)` labels whatever the cap left out.
  paintRows(items, cap, box, moreId, moreText, toRow){
    const kept = new Map();
    items.slice(0, cap).forEach(item => {
      const {id, sig, data} = toRow(item);
      const had = this._rows.get(id);
      const row = had ? had.row : document.createElement("list-row");
      // Left alone when nothing it would say has changed: rewriting it drops
      // the focus off its disclosure button and closes it if it was open.
      if(!had || had.sig !== sig){
        row.data = {...data, open: id === this._openId,
                    onToggle: open => this.opened(open ? id : null)};
      }
      kept.set(id, {row, sig});
    });
    // Gone first, then placed: a row put in position ahead of a row that is
    // about to be removed would be moved twice, and moving one is what takes
    // the focus out of it.
    this._rows.forEach(({row}, id) => { if(!kept.has(id)) row.remove(); });
    this._rows = kept;
    [...kept.values()].forEach(({row}, at) => {
      const there = box.children[at];
      if(there !== row) box.insertBefore(row, there || null);
    });
    if(!this._rows.has(this._openId)) this._openId = null;

    const rest = items.length - cap;
    let note = this.$(`#${moreId}`);
    if(rest > 0 && !note){
      note = document.createElement("p");
      note.id = moreId;
      note.className = "hint";
      box.appendChild(note);
    }
    if(rest > 0) note.textContent = moreText(rest);
    else if(note) note.remove();
  }

  // One at a time: several open at once turns a list of six into a page of
  // detail with no list left in it.
  opened(id){
    this._openId = id;
    this._rows.forEach(({row}, key) => { if(key !== id) row.setOpen(false); });
  }
}
