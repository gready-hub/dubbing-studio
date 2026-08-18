export async function withBusy(buttons, label, fn){
  const list = Array.isArray(buttons) ? buttons : [buttons];
  const originals = list.map(b => b.textContent);
  list.forEach(b => {
    b.disabled = true;
    if(label) b.textContent = label;
  });
  try{
    return await fn();
  } finally {
    list.forEach((b, i) => {
      b.disabled = false;
      if(label) b.textContent = originals[i];
    });
  }
}
