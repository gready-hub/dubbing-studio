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

export function errorDetailHtml(message, detail){
  const trimmed = (detail || "").trim();
  if(!trimmed || trimmed === message) return escapeHtml(message);
  return `${escapeHtml(message)}<details style="margin-top:8px">
    <summary>Details</summary>
    <pre>${escapeHtml(trimmed)}</pre>
  </details>`;
}
