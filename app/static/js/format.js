export function fmt(sec){
  sec = Math.max(0, Math.round(sec||0));
  if(sec < 60) return sec+"s";
  const m = Math.floor(sec/60), s = sec%60;
  // "15m 0s" spends a word on nothing; the seconds are only worth printing when
  // there are some.
  if(m < 60) return s ? `${m}m ${s}s` : `${m}m`;
  return `${Math.floor(m/60)}h ${m%60}m`;
}

// Deliberately coarser than fmt(). An estimate given to the second claims a
// precision it does not have, and "about 41m 12s left" invites the reader to
// hold it to that.
export function fmtRough(sec){
  sec = Math.max(0, Math.round(sec||0));
  if(sec < 90) return "a minute";
  if(sec < 3600) return `${Math.round(sec/60)} min`;
  const h = Math.floor(sec/3600), m = Math.round((sec%3600)/60);
  return m ? `${h}h ${m}m` : `${h}h`;
}

export const fmtBytes = b => b >= 1073741824 ? (b/1073741824).toFixed(1)+" GB"
                    : b >= 1048576    ? Math.round(b/1048576)+" MB"
                    : Math.round(b/1024)+" KB";

export const escapeHtml = s => String(s??"").replace(/[&<>"']/g,
  c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
export const escapeAttr = s => escapeHtml(s).replace(/'/g,"&#39;");

// "2026-08-16" -> "16 August". Nothing turns on it being exact, so a value that
// will not parse is simply left out rather than shown as Invalid Date.
export function niceDate(iso){
  const d = new Date(iso + "T00:00:00");
  return isNaN(d) ? iso
    : d.toLocaleDateString(undefined, {day:"numeric", month:"long"});
}

// "/Users/x/Movies/Dubbed" -> "Movies → Dubbed". The whole path means nothing
// to anyone; the last two parts are what they will recognise in Finder.
export function friendlyFolder(path){
  const parts = String(path || "").split("/").filter(Boolean);
  return parts.slice(-2).join(" → ") || "your Movies folder";
}

// An epoch-seconds stamp -> the local day and clock, kept apart so a caller
// can lay them out however it needs. Built from local date parts rather than
// read off the UTC day, or an evening job would land on the day after; null
// when there is no stamp to show.
export function dayAndClock(epochSeconds){
  const at = new Date((epochSeconds || 0) * 1000);
  if(!epochSeconds || isNaN(at)) return null;
  const pad = n => String(n).padStart(2, "0");
  const day = niceDate(`${at.getFullYear()}-${pad(at.getMonth()+1)}-${pad(at.getDate())}`);
  const year = at.getFullYear() === new Date().getFullYear() ? "" : ` ${at.getFullYear()}`;
  return {day: `${day}${year}`,
          clock: at.toLocaleTimeString(undefined, {hour: "numeric", minute: "2-digit"})};
}

export const dayAndClockText = epochSeconds => {
  const at = dayAndClock(epochSeconds);
  return at ? `${at.day} at ${at.clock}` : "";
};

// A URL as a clickable link with the scheme and "www." trimmed off what's
// shown, or as plain escaped text when it isn't actually a link.
export function link(url){
  const shown = escapeHtml(String(url).replace(/^https?:\/\/(www\.)?/i, ""));
  return /^https?:\/\//i.test(url)
    ? `<a href="${escapeAttr(url)}" target="_blank" rel="noopener"
         style="color:var(--accent)">${shown}</a>`
    : shown;
}

export function errorDetailHtml(message, detail){
  const trimmed = (detail || "").trim();
  if(!trimmed || trimmed === message) return escapeHtml(message);
  return `${escapeHtml(message)}<details style="margin-top:8px">
    <summary>Details</summary>
    <pre>${escapeHtml(trimmed)}</pre>
  </details>`;
}

// A message put on an element and taken off again after `ms`. A later message
// on the same element cancels the pending clear, so the last one written is the
// one that stands; `ms` of 0 leaves it there until something replaces it.
export function flash(el, text, ms = 1800){
  clearTimeout(el._flashTimer);
  el.textContent = text;
  if(ms) el._flashTimer = setTimeout(() => { el.textContent = ""; }, ms);
}
