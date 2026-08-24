// Short names for the chip row. Which of them a job shows comes from the job
// itself, since that depends on the preset — this list used to be fixed and had
// no entry for separation or speaker detection, so nothing was highlighted for
// the whole of Demucs and the row looked dead at the very moment the user is
// wondering whether anything is happening.
//
// Named for the work itself, matching ALL_STAGES in pipeline.py: "Listen" and
// "Speak" gave a progress chip more character than the step behind it.
export const STAGE_LABELS = {
  download:"Download", separate:"Separate", diarize:"Speakers",
  transcribe:"Transcribe", translate:"Translate", synthesize:"Speech",
  assemble:"Align", finish:"Save"
};
// "Download" is the wrong word for a file that is already on this Mac. It is
// the same slot in the same plan either way — see JobRunner._plan — and what it
// does in that slot is open the file rather than fetch it.
export const stageLabel = (key, job) =>
  (key === "download" && job && job.local) ? "Open" : (STAGE_LABELS[key] || key);

// Until the job reports its own plan, which it does as soon as it starts.
export const DEFAULT_STAGES = ["download","transcribe","translate","synthesize","assemble","finish"];
