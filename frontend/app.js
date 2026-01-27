const CONFIG = {
  API_BASE: 'http://localhost:8000'
};

let currentRepoId = null;
let statusPollInterval = null;
let isRepoReady = false;

const repoUrlInput = document.getElementById('repoUrl');
const branchInput = document.getElementById('branchInput');
const ingestBtn = document.getElementById('ingestBtn');
const statusPanel = document.getElementById('statusPanel');
const chatMessages = document.getElementById('chatMessages');
const questionInput = document.getElementById('questionInput');
const askBtn = document.getElementById('askBtn');

ingestBtn.addEventListener('click', handleIngest);
askBtn.addEventListener('click', handleAsk);
questionInput.addEventListener('keypress', (e) => {
  if (e.key === 'Enter' && e.ctrlKey && !askBtn.disabled) {
    handleAsk();
  }
});

function extractRepoId(url) {
  try {
    const urlObj = new URL(url);
    const parts = urlObj.pathname.split('/').filter(Boolean);
    if (parts.length >= 2) {
      const owner = parts[0];
      const repo = parts[1].replace('.git', '');
      return `${owner}/${repo}`;
    }
  } catch (e) {
    return null;
  }
  return null;
}

async function handleIngest() {
  const url = repoUrlInput.value.trim();
  const branch = branchInput.value.trim() || 'main';

  if (!url) {
    showError('Please enter a GitHub URL');
    return;
  }

  const repoId = extractRepoId(url);
  if (!repoId) {
    showError('Invalid GitHub URL format');
    return;
  }

  currentRepoId = repoId;
  isRepoReady = false;

  ingestBtn.disabled = true;
  repoUrlInput.disabled = true;
  branchInput.disabled = true;
  questionInput.disabled = true;
  askBtn.disabled = true;

  updateStatusPanel('processing', 'Sending ingestion request...');

  try {
    const response = await fetch(`${CONFIG.API_BASE}/ingest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ github_url: url, branch })
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ message: response.statusText }));
      throw new Error(error.message || `HTTP ${response.status}`);
    }

    const data = await response.json();
    currentRepoId = data.repo_id;

    updateStatusPanel('processing', `Ingestion started. Repo ID: ${data.repo_id}`);

    clearInterval(statusPollInterval);
    statusPollInterval = setInterval(() => pollStatus(data.repo_id), 2000);
    pollStatus(data.repo_id);
  } catch (error) {
    console.error('Ingest error:', error);
    updateStatusPanel('failed', `Error: ${error.message}`);
    resetIngestForm();
  }
}

async function pollStatus(repoId) {
  try {
    const response = await fetch(`${CONFIG.API_BASE}/status/${encodeURIComponent(repoId)}`);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();

    if (data.status === 'ready') {
      clearInterval(statusPollInterval);
      isRepoReady = true;

      updateStatusPanel('ready', `Repository ready for querying (${data.commits_indexed} commits indexed)`);

      questionInput.disabled = false;
      askBtn.disabled = false;

      chatMessages.innerHTML = '<div class="empty-state">Ask a question about the repository.</div>';
    } else if (data.status === 'error') {
      clearInterval(statusPollInterval);
      updateStatusPanel('failed', `Failed: ${data.error || 'Unknown error'}`);
      resetIngestForm();
    } else {
      updateStatusPanel(
        'processing',
        `${data.status || 'Processing'}${data.commits_indexed ? ` (${data.commits_indexed} commits)` : ''}`
      );
    }
  } catch (error) {
    console.error('Status poll error:', error);
    updateStatusPanel('failed', 'Status check failed');
  }
}

async function handleAsk() {
  const question = questionInput.value.trim();

  if (!question || !isRepoReady) return;

  addChatMessage('user', question);
  questionInput.value = '';
  questionInput.focus();

  askBtn.disabled = true;
  questionInput.disabled = true;

  try {
    const response = await fetch(`${CONFIG.API_BASE}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, repo_id: currentRepoId })
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ message: response.statusText }));
      throw new Error(error.message || `HTTP ${response.status}`);
    }

    const data = await response.json();
    addChatMessage('assistant', data.answer, data.citations);
  } catch (error) {
    console.error('Query error:', error);
    addChatMessage('assistant', `Error: ${error.message}`);
  } finally {
    askBtn.disabled = false;
    questionInput.disabled = false;
    questionInput.focus();
  }
}

function addChatMessage(role, content, citations = null) {
  const chatContainer = chatMessages;

  const emptyState = chatContainer.querySelector('.empty-state');
  if (emptyState) emptyState.remove();

  const messageDiv = document.createElement('div');
  messageDiv.className = `message ${role}`;

  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';
  bubble.textContent = content;
  messageDiv.appendChild(bubble);

  if (Array.isArray(citations) && citations.length > 0) {
    const citationsDiv = document.createElement('div');
    citationsDiv.className = 'citations';
    citationsDiv.innerHTML = '<strong>Sources:</strong>';

    citations.forEach((c) => {
      const pill = document.createElement('span');
      pill.className = 'citation-link';

      const commit = (c.commit || '').slice(0, 7);
      const author = c.author || 'Unknown';
      const date = c.date || '';
      const files = Array.isArray(c.files) && c.files.length
        ? ` • ${c.files.slice(0, 3).join(', ')}${c.files.length > 3 ? '…' : ''}`
        : '';

      pill.textContent = `${commit} • ${author} • ${date}${files}`;
      citationsDiv.appendChild(pill);
    });

    messageDiv.appendChild(citationsDiv);
  }

  chatContainer.appendChild(messageDiv);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function updateStatusPanel(status, message) {
  let statusClass = '';
  let icon = '';

  switch (status) {
    case 'ready':
      statusClass = 'status-ready';
      icon = '✓';
      break;
    case 'failed':
      statusClass = 'status-failed';
      icon = '✗';
      break;
    case 'processing':
      statusClass = 'status-processing';
      icon = '<span class="spinner"></span>';
      break;
  }

  statusPanel.innerHTML = `
    <div class="status-item">
      <span class="status-label">${icon} Status:</span>
      <span class="status-value ${statusClass}">${message}</span>
    </div>
    <div class="status-item">
      <span class="status-label">Repo ID:</span>
      <span class="status-value">${currentRepoId || 'None'}</span>
    </div>
  `;
}

function showError(message) {
  const errorDiv = document.createElement('div');
  errorDiv.className = 'error-message';
  errorDiv.textContent = message;

  const parent = statusPanel.parentElement;
  parent.insertBefore(errorDiv, parent.firstChild);

  setTimeout(() => errorDiv.remove(), 5000);
}

function resetIngestForm() {
  ingestBtn.disabled = false;
  repoUrlInput.disabled = false;
  branchInput.disabled = false;
  currentRepoId = null;
  isRepoReady = false;
}

console.log(`Repo-Trace frontend initialized. API Base: ${CONFIG.API_BASE}`);
