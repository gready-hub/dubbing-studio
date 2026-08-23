import { api } from "./api.js";
import { escapeHtml, fmt } from "./format.js";
import { STAGE_LABELS } from "./stages.js";

// A sample lives in the working files, which get tidied up; a finished video can
// be moved or deleted from the videos folder while the app is not looking.
export const SAMPLE_GONE = "The sample file has been deleted — samples are stored "
  + "with the working files. Run it again to make a new one.";
export const OUTPUT_GONE = "The file is no longer there. It may have been moved, "
  + "renamed or deleted.";

// The words the controls ask these choices in, so a finished run is described in
// the terms it was asked for. Every one of them has an <option> or a button in
// settings-panel.js or new-job-panel.js that must say the same thing.
const VOICE_MODES = {fixed: "Use a built-in voice", clone: "Clone the original speaker"};
const AUDIO_MODES = {replace: "Replace completely", duck: "Keep quietly underneath",
                     dual: "Keep as a second track"};
const DUCK_LEVELS = {"-12": "quiet (-12 dB)", "-18": "very quiet (-18 dB)",
                     "-24": "barely there (-24 dB)"};
const SPEEDS = {0.9: "Slower", 1: "Normal", 1.1: "Slightly faster"};
const SEPARATION = {true: "Separate and keep them", false: "Don't separate (faster)"};
const MUSIC_BED = {true: "Mix it back under the new voices", false: "Drop it — voices only"};
const TRANSLATORS = {ollama: "Local model (free, private)",
                     anthropic: "Claude API (best quality)", openai: "OpenAI API"};
const MODEL_LABELS = {ollama: "Local model", anthropic: "Claude model", openai: "OpenAI model"};
// The same words the buttons on the front page carry.
const SPEAKING = {true: "Several people", false: "One person"};
const ASR_MODELS = {parakeet: "Parakeet — fast", whisper: "Whisper — more accurate, slower"};
const VIDEO_QUALITY = {best: "Best available", 1080: "Up to 1080p",
                       720: "Up to 720p — much smaller"};
const SUBTITLES = {true: "Also save an .srt", false: "No subtitle file"};
const RUN_ON_LINES = {true: "Join lines that run together", false: "Keep them exactly as heard"};
const MAX_STRETCH = {1.3: "Gentle — 1.3x", 1.55: "Normal — 1.55x", 1.8: "Firm — 1.8x"};

// One choice, in the words of the single control that asks it: the mode, and
// with it the level the original is held at where there is one. Only duck mode
// has a level — replacing the original leaves nothing to hold down, and a second
// track is played at whatever volume the viewer gives it.
function audioChoice(snap){
  const mode = AUDIO_MODES[snap.audio_mode] ?? String(snap.audio_mode);
  if(snap.audio_mode !== "duck" || snap.duck_db === undefined) return mode;
  return `${mode} — ${DUCK_LEVELS[snap.duck_db] ?? `${snap.duck_db} dB`}`;
}

function people(n){
  if(Number(n) < 0) return "Work it out automatically";
  return Number(n) === 1 ? "1 person" : `${n} people`;
}

// A preset name, in the same words the quality picker uses — falling back to
// a capitalised raw value for a preset the shipped list has since dropped.
export function presetLabel(preset, presets){
  const found = (presets || {})[preset];
  return found ? found.label
    : String(preset).charAt(0).toUpperCase() + String(preset).slice(1);
}

// A finished run described as a list of [label, value] pairs. With `outcomes`,
// what the run made of those choices is appended as well — for a caller with
// nowhere else to put the results. A panel that reports them separately asks for
// the choices alone.
export function runRows(job, state, {outcomes = false} = {}){
  const stats = job.stats || {};
  const snap = stats.settings || {};
  const rows = [];
  // Runs from before the app recorded the settings it ran under have no
  // snapshot, so every one of these has to be allowed to be missing.
  const add = (key, label, render) => {
    if(snap[key] !== undefined) rows.push([label, render(snap[key])]);
  };
  const from = (map, fallback = String) => v => map[v] ?? fallback(v);

  // Those runs still named their preset, out alongside the results. "custom" is
  // one of its values and is on no list of presets, so an unrecognised name is
  // shown rather than dropped.
  const preset = snap.preset !== undefined ? snap.preset : stats.preset;
  if(preset) rows.push(["Quality preset", presetLabel(preset, state.presets)]);
  add("voice_mode", "Cloning", from(VOICE_MODES));
  // The chosen built-in voice had no part in a run that cloned the original
  // speakers; what did the speaking is a result rather than a choice.
  if(snap.voice_mode !== "clone"){
    add("voice", "Built-in voice",
        v => ((state.voices || []).find(o => o.id === v) || {}).label || v);
  }
  add("speed", "Speaking speed", from(SPEEDS, v => `${v}x`));
  add("audio_mode", "Original audio", () => audioChoice(snap));
  add("separate_audio", "Music and effects", from(SEPARATION));
  // Only recorded when it had a bearing on the run — there is music to put back
  // only when speech was separated from it, and room for it underneath only
  // when the original audio is being replaced. See Settings.run_snapshot().
  add("keep_music", "The separated music and effects", from(MUSIC_BED));
  add("translator", "Translation", from(TRANSLATORS));
  // The model asked for, which is not always the one that answered: a local
  // model left empty is chosen per machine, and stats.translated_by is the one
  // that did the work.
  if(MODEL_LABELS[snap.translator] && (snap.translator_model || snap.translator === "ollama")){
    add("translator_model", MODEL_LABELS[snap.translator], v => v || "Chosen automatically");
  }
  add("target_language", "Translate into", String);
  add("glossary", "Built-in terms", v => (state.glossaries || {})[v] || v);
  if(snap.has_custom_glossary === true) rows.push(["Your own terms", "Used"]);
  add("diarize", "Who's speaking", from(SPEAKING));
  if(snap.diarize === true) add("expected_speakers", "How many people speak", people);
  add("asr_model", "Transcription engine", from(ASR_MODELS));
  add("keep_video_quality", "Video quality", from(VIDEO_QUALITY));
  add("write_srt", "Subtitles", from(SUBTITLES));
  add("merge_lines", "Run-on lines", from(RUN_ON_LINES));
  add("max_stretch", "Hardest allowed squeeze", from(MAX_STRETCH, v => `${v}x`));

  if(outcomes){
    if(stats.speakers) rows.push(["Speakers found", String(stats.speakers)]);
    if(stats.translated_by) rows.push(["Translated by", stats.translated_by]);
    if(stats.voices) rows.push(["Voices used", stats.voices]);
    if(stats.engine && stats.engine !== stats.voices) rows.push(["Engine", stats.engine]);
  }
  return rows;
}

export const statRows = rows => rows.map(([k, v]) =>
  `<div class="stat"><span>${escapeHtml(k)}</span><span>${escapeHtml(v)}</span></div>`).join("");

// Longest first, and only the stages that took real time.
export const stages = job => Object.entries(job.stage_times || {})
  .filter(([,sec]) => sec >= 1)
  .sort((a,b) => b[1] - a[1])
  .map(([key,sec]) => `<div class="stat sub"><span>`
    + `${escapeHtml(STAGE_LABELS[key] || key)}</span>`
    + `<span>${escapeHtml(fmt(sec))}</span></div>`).join("");

// output_exists is worked out afresh every time the server is asked, and the
// browser asks once: /api/state at boot, and after that only live jobs arrive.
// A video deleted while the app is open is still listed as being there, and
// /api/reveal on a path that has gone opens nothing and reports nothing — so
// the ones found to have gone are remembered here.
const goneOutputs = new Set();

export const knownGone = path => goneOutputs.has(path);

export async function outputGone(path){
  if(!path) return true;
  if(goneOutputs.has(path)) return true;
  try{
    const entry = ((await api.state()).jobs || []).find(j => j.output === path);
    if(entry && entry.output_exists === false){
      goneOutputs.add(path);
      return true;
    }
  }catch(err){
    // A check that cannot be made is not an answer: go ahead and open it.
  }
  return false;
}
