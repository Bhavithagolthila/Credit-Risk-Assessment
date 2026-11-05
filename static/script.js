const form = document.getElementById('assessmentForm');
const submitButton = document.getElementById('submitButton');
const formError = document.getElementById('formError');
const incomeInput = form.elements.person_income;
const amountInput = document.getElementById('loanAmount');
const ratioInput = document.getElementById('loanRatio');

function updateRatio() {
  const income = Number(incomeInput.value);
  const amount = Number(amountInput.value);
  ratioInput.value = income > 0 && amount >= 0 ? (amount / income).toFixed(2) : '0.00';
}
const slider = document.getElementById('loanSlider');
function paintSlider() {
  const min = Number(slider.min), max = Number(slider.max);
  slider.style.setProperty('--fill', `${((slider.value - min) / (max - min)) * 100}%`);
}
slider.addEventListener('input', () => { amountInput.value = slider.value; paintSlider(); updateRatio(); });
amountInput.addEventListener('input', () => { slider.value = amountInput.value || slider.min; paintSlider(); });
incomeInput.addEventListener('input', updateRatio);
amountInput.addEventListener('input', updateRatio);
updateRatio();
paintSlider();

function setService(online, label) {
  document.getElementById('statusDot').className = `status-dot ${online ? 'online' : 'offline'}`;
  document.getElementById('serviceStatus').textContent = label;
}

async function checkService() {
  try {
    const response = await fetch('/health');
    const data = await response.json();
    setService(response.ok && data.model_loaded, response.ok && data.model_loaded ? 'Service ready' : 'Model unavailable');
  } catch (_) {
    setService(false, 'Service offline');
  }
}

function percent(value) { return `${(Number(value) * 100).toFixed(1)}%`; }

function showResult(data) {
  document.getElementById('emptyResult').classList.add('hidden');
  document.getElementById('resultContent').classList.remove('hidden');
  const high = data.default_prediction === 1;
  const title = document.getElementById('riskTitle');
  title.textContent = data.result;
  document.getElementById('verdictPill').classList.toggle('high', high);
  document.getElementById('defaultProbability').parentElement.classList.toggle('high', high);
  document.getElementById('riskDescription').textContent = high
    ? 'The estimated default probability is at or above the model decision threshold.'
    : 'The estimated default probability is below the model decision threshold.';
  document.getElementById('defaultProbability').textContent = percent(data.default_probability);
  document.getElementById('repaymentProbability').textContent = percent(data.repayment_probability);
  document.getElementById('thresholdStat').textContent = percent(data.threshold);
  document.getElementById('thresholdValue').textContent = percent(data.threshold);
  document.getElementById('resultDisclaimer').textContent = data.disclaimer;
  const pin = document.getElementById('probabilityFill');
  pin.style.left = '0%';
  requestAnimationFrame(() => requestAnimationFrame(() => {
    pin.style.left = `${Math.max(0, Math.min(100, data.default_probability * 100))}%`;
  }));
  document.getElementById('thresholdMarker').style.left = `${data.threshold * 100}%`;
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  formError.textContent = '';
  if (!form.reportValidity()) return;
  updateRatio();
  const payload = Object.fromEntries(new FormData(form).entries());
  for (const key of ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']) {
    payload[key] = Number(payload[key]);
  }
  submitButton.disabled = true;
  submitButton.querySelector('span:first-child').textContent = 'Checking…';
  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (!response.ok) {
      const detail = Array.isArray(data.detail) ? data.detail.map(item => item.msg).join(' ') : data.detail;
      throw new Error(detail || 'The assessment could not be completed.');
    }
    showResult(data);
  } catch (error) {
    formError.textContent = error.message || 'Unable to connect to the assessment service.';
  } finally {
    submitButton.disabled = false;
    submitButton.querySelector('span:first-child').textContent = 'Check risk';
  }
});

document.getElementById('resetButton').addEventListener('click', () => {
  form.reset();
  paintSlider();
  updateRatio();
  document.getElementById('resultContent').classList.add('hidden');
  document.getElementById('emptyResult').classList.remove('hidden');
  formError.textContent = '';
  window.scrollTo({ top: document.getElementById('formTitle').offsetTop - 100, behavior: 'smooth' });
});

async function loadModelInfo() {
  const container = document.getElementById('importanceList');
  try {
    const response = await fetch('/model-info');
    if (!response.ok) throw new Error('Model information is unavailable.');
    const data = await response.json();
    if (!data.global_feature_importance || !data.global_feature_importance.length) {
      container.innerHTML = '<span class="loading-line">Global importance is not available for this saved model.</span>';
      return;
    }
    const max = Math.max(...data.global_feature_importance.map(item => item.importance));
    container.innerHTML = data.global_feature_importance.map(item => `
      <div class="importance-row">
        <span title="${item.feature}">${item.feature}</span>
        <div class="importance-bar"><i style="width:${Math.max(2, item.importance / max * 100)}%"></i></div>
        <small>${(item.importance * 100).toFixed(1)}%</small>
      </div>`).join('');
  } catch (_) {
    container.innerHTML = '<span class="loading-line">Connect to the API to view model information.</span>';
  }
}

checkService();
loadModelInfo();
