class Store {
  constructor(){
    this.state = {
      settings: {}, settings_path: "", jobs: {}, current: null,
      features: {}, presets: {}, glossaries: {}, voices: [],
      machine: {}, output_dir: "", storage: null, doctor: null,
    };
    this._subs = new Set();
  }

  subscribe(fn){
    this._subs.add(fn);
    fn(this.state);
    return () => this._subs.delete(fn);
  }

  notify(){
    for(const fn of this._subs) fn(this.state);
  }

  set(patch){
    Object.assign(this.state, patch);
    this.notify();
  }

  setJob(job){
    this.state.jobs = {...this.state.jobs, [job.id]: job};
    this.notify();
  }

  setCurrent(id){
    this.state.current = id;
    this.notify();
  }

  setSettings(settings){
    this.state.settings = settings;
    this.notify();
  }

  setStorage(storage){
    this.state.storage = storage;
    this.notify();
  }

  setDoctor(doctor){
    this.state.doctor = doctor;
    this.notify();
  }
}

export const store = new Store();
