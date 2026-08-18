// Short names for the chip row. Which of them a job shows comes from the job
// itself, since that depends on the preset — this list used to be fixed and had
// no entry for separation or speaker detection, so nothing was highlighted for
// the whole of Demucs and the row looked dead at the very moment the user is
// wondering whether anything is happening.
export const STAGE_LABELS = {
  download:"Download", separate:"Separate", diarize:"Speakers",
  transcribe:"Listen", translate:"Translate", synthesize:"Speak",
  assemble:"Fit", finish:"Save"
};
// Until the job reports its own plan, which it does as soon as it starts.
export const DEFAULT_STAGES = ["download","transcribe","translate","synthesize","assemble","finish"];
