(function(){
  const PREDICT_ENDPOINT = '/api/predict'; // Change to your backend endpoint

  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const browseBtn = document.getElementById('browse-btn');
  const predictBtn = document.getElementById('predict-btn');
  const clearBtn = document.getElementById('clear-btn');
  const preview = document.getElementById('preview');
  const previewImg = document.getElementById('preview-img');
  const resultCard = document.getElementById('result-card');
  const resultBadge = document.getElementById('result-badge');
  const progressBar = document.getElementById('progress-bar');
  const confidenceText = document.getElementById('confidence-text');
  const explanation = document.getElementById('explanation');
  const toast = document.getElementById('toast');

  let currentFile = null;

  function showToast(msg, isError){
    toast.textContent = msg;
    toast.style.borderColor = isError ? 'hsl(var(--danger))' : 'hsl(var(--border))';
    toast.classList.add('show');
    setTimeout(()=> toast.classList.remove('show'), 2500);
  }

  function setActiveDrag(active){
    dropzone.setAttribute('data-active', active ? 'true' : 'false');
  }

  function reset(){
    currentFile = null;
    previewImg.src = '';
    preview.classList.add('hidden');
    document.getElementById('dropzone-empty').classList.remove('hidden');
    predictBtn.disabled = true;
    clearBtn.disabled = true;
    resultCard.classList.add('hidden');
  }

  function handleFile(file){
    if(!file || !file.type || !file.type.startsWith('image/')){
      showToast('Please upload a valid image file.', true);
      return;
    }
    currentFile = file;
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    preview.classList.remove('hidden');
    document.getElementById('dropzone-empty').classList.add('hidden');
    predictBtn.disabled = false;
    clearBtn.disabled = false;
  }

  // Drag and drop events
  ['dragenter','dragover'].forEach(evt => {
    dropzone.addEventListener(evt, (e)=>{ e.preventDefault(); e.stopPropagation(); setActiveDrag(true); });
  });
  ;['dragleave','drop'].forEach(evt => {
    dropzone.addEventListener(evt, (e)=>{ e.preventDefault(); e.stopPropagation(); setActiveDrag(false); });
  });
  dropzone.addEventListener('drop', (e)=>{
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if(file) handleFile(file);
  });

  dropzone.addEventListener('keydown', (e)=>{
    if(e.key === 'Enter' || e.key === ' '){
      e.preventDefault();
      fileInput.click();
    }
  });

  browseBtn.addEventListener('click', ()=> fileInput.click());
  fileInput.addEventListener('change', (e)=>{
    const file = e.target.files && e.target.files[0];
    if(file) handleFile(file);
  });

  clearBtn.addEventListener('click', reset);

  predictBtn.addEventListener('click', async ()=>{
    if(!currentFile) return;
    predictBtn.disabled = true;
    predictBtn.classList.add('loading');
    predictBtn.querySelector('.btn__label').textContent = 'Predicting...';
    progressBar.classList.add('animated');
    resultCard.classList.add('hidden');

    try{
      const form = new FormData();
      form.append('file', currentFile);
      const res = await fetch(PREDICT_ENDPOINT, { method:'POST', body: form });
      if(!res.ok) throw new Error(`Prediction failed (${res.status})`);
      const data = await res.json();

      let label = String(data.label || 'Unknown');
      let confidence = Number(data.confidence || 0);
      if(confidence > 1) confidence = confidence / 100;
      const pct = Math.round(confidence * 100);

      resultBadge.textContent = label;
      resultBadge.style.borderColor = 'hsl(var(--border))';
      if(/ai/i.test(label)){
        resultBadge.style.background = 'hsl(var(--danger) / .1)';
      }else{
        resultBadge.style.background = 'hsl(var(--ring) / .12)';
      }

      confidenceText.textContent = pct + '%';
      progressBar.style.width = pct + '%';
      explanation.textContent = data.explanation ? String(data.explanation) : '';

      resultCard.classList.remove('hidden');
      showToast('Prediction complete');
    }catch(err){
      console.error(err);
      showToast(err.message || 'Failed to predict', true);
    }finally{
      predictBtn.disabled = false;
      predictBtn.classList.remove('loading');
      predictBtn.querySelector('.btn__label').textContent = 'Predict';
      progressBar.classList.remove('animated');
    }
  });

})();
