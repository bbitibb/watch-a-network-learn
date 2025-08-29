const API          = "http://127.0.0.1:8000";   // NEW
const uploadForm   = document.getElementById('uploadForm');
const imgInput     = document.getElementById('imgInput');
const statusDiv    = document.getElementById('status');
const epochSlider  = document.getElementById('epochSlider');
const epochLabel   = document.getElementById('epochLabel');
const reconImg     = document.getElementById('reconImg');
const lossDisplay  = document.getElementById('lossDisplay');
const origImg      = document.getElementById('origImg');

let numEpochs = 0;
let losses    = [];

uploadForm.onsubmit = async (e) => {
  e.preventDefault();
  const file = imgInput.files[0];
  if (!file) return;

  // preview
  origImg.src = URL.createObjectURL(file);

  const formData = new FormData();
  formData.append("file", file);

  statusDiv.textContent = "Uploading & training…";
  await fetch(`${API}/upload`, { method: "POST", body: formData });
  statusDiv.textContent = "Training in progress…";
  setTimeout(init, 2000);          // start polling
};

async function init() {
  const { num_epochs } = await fetch(`${API}/epochs`).then(r => r.json());
  if (num_epochs === 0) return setTimeout(init, 2000);

  numEpochs = num_epochs;
  epochSlider.min  = 1;
  epochSlider.max  = numEpochs;
  epochSlider.value = numEpochs;
  epochLabel.textContent = `Epoch: ${numEpochs}/${numEpochs}`;

  await fetchLosses();
  showReconstruction(numEpochs);
  epochSlider.disabled = false;
}

epochSlider.oninput = () => {
  const epoch = parseInt(epochSlider.value);
  epochLabel.textContent = `Epoch: ${epoch}/${numEpochs}`;
  showReconstruction(epoch);
};

function showReconstruction(epoch) {
  reconImg.src = `${API}/reconstruction/${epoch}?t=${Date.now()}`; // cache-buster
  if (losses.length >= epoch) lossDisplay.textContent =
    `Loss: ${losses[epoch-1].toFixed(5)}`;
}

async function fetchLosses() {
  const data = await fetch(`${API}/losses`).then(r => r.json());
  losses = data.losses ?? [];
  showReconstruction(parseInt(epochSlider.value));
}

// poll every 5 s for new epochs
setInterval(async () => {
  const { num_epochs } = await fetch(`${API}/epochs`).then(r => r.json());
  if (num_epochs > numEpochs) {
    numEpochs = num_epochs;
    epochSlider.max  = numEpochs;
    await fetchLosses();
  }
}, 5000);
