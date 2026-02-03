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
const themeToggle = document.getElementById('themeToggle');

ingestBtn.addEventListener('click', handleIngest);
askBtn.addEventListener('click', handleAsk);
themeToggle.addEventListener('click', toggleTheme);
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

      updateStatusPanel('ready', `Repository ready for querying (${data.commits_indexed} commits indexed)`, {
        cache_hits: data.cache_hits || 0,
        cache_misses: data.cache_misses || 0,
        cache_hit_rate: data.cache_hit_rate || 0
      });

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
        `${data.status || 'Processing'}${data.commits_indexed ? ` (${data.commits_indexed} commits)` : ''}`,
        data.cache_hits !== undefined ? {
          cache_hits: data.cache_hits,
          cache_misses: data.cache_misses,
          cache_hit_rate: data.cache_hit_rate
        } : null
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
    addChatMessage('assistant', data.answer, data.citations, data.referenced_files);
  } catch (error) {
    console.error('Query error:', error);
    addChatMessage('assistant', `Error: ${error.message}`);
  } finally {
    askBtn.disabled = false;
    questionInput.disabled = false;
    questionInput.focus();
  }
}

function processMarkdownContent(content) {
  // First escape HTML entities
  let processed = content
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  // Process code blocks ```language\ncode\n```
  processed = processed.replace(/```(\w+)?\n?([\s\S]*?)```/g, (match, language, code) => {
    const lang = language || 'plaintext';
    const langDisplay = language ? language.charAt(0).toUpperCase() + language.slice(1) : 'Code';
    
    return `<div class="code-block">
      <div class="code-header">
        <span>${langDisplay}</span>
        <button class="copy-btn" onclick="copyCode(this)">Copy</button>
      </div>
      <div class="code-content">
        <pre><code class="language-${lang}">${code.trim()}</code></pre>
      </div>
    </div>`;
  });

  // Process inline code `code`
  processed = processed.replace(/`([^`\n]+)`/g, '<code style="background: rgba(0,0,0,0.2); padding: 2px 4px; border-radius: 3px; font-family: monospace;">$1</code>');

  // Process other markdown
  processed = processed
    .replace(/\n/g, '<br>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>');

  return processed;
}

function copyCode(button) {
  const codeBlock = button.closest('.code-block');
  const codeElement = codeBlock.querySelector('code');
  const code = codeElement.textContent;
  
  navigator.clipboard.writeText(code).then(() => {
    button.textContent = 'Copied!';
    button.classList.add('copied');
    setTimeout(() => {
      button.textContent = 'Copy';
      button.classList.remove('copied');
    }, 2000);
  }).catch(err => {
    console.error('Failed to copy code:', err);
    button.textContent = 'Error';
    setTimeout(() => {
      button.textContent = 'Copy';
    }, 2000);
  });
}

function addChatMessage(role, content, citations = null, referencedFiles = null) {
  const chatContainer = chatMessages;

  const emptyState = chatContainer.querySelector('.empty-state');
  if (emptyState) emptyState.remove();

  const messageDiv = document.createElement('div');
  messageDiv.className = `message ${role}`;

  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';
  
  // Process content with markdown code blocks
  const processedContent = processMarkdownContent(content);
  bubble.innerHTML = processedContent;
  
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

  // Add referenced files sidebar if available
  if (Array.isArray(referencedFiles) && referencedFiles.length > 0) {
    const filesSidebar = document.createElement('div');
    filesSidebar.className = 'referenced-files-sidebar';
    
    const toggleBtn = document.createElement('button');
    toggleBtn.className = 'files-toggle-btn';
    toggleBtn.innerHTML = '📁 Referenced Files <span class="toggle-icon">▼</span>';
    toggleBtn.onclick = () => {
      const content = filesSidebar.querySelector('.files-content');
      const icon = filesSidebar.querySelector('.toggle-icon');
      if (content.style.display === 'none') {
        content.style.display = 'block';
        icon.textContent = '▼';
      } else {
        content.style.display = 'none';
        icon.textContent = '▶';
      }
    };
    
    const filesContent = document.createElement('div');
    filesContent.className = 'files-content';
    
    referencedFiles.forEach((file) => {
      const fileItem = document.createElement('div');
      fileItem.className = 'file-item';
      
      const fileHeader = document.createElement('div');
      fileHeader.className = 'file-header';
      
      const fileName = document.createElement('span');
      fileName.className = 'file-name';
      fileName.textContent = file.file_path.split('/').pop();
      
      const relevance = document.createElement('span');
      relevance.className = 'file-relevance';
      relevance.textContent = `${(file.relevance_score * 100).toFixed(1)}%`;
      
      fileHeader.appendChild(fileName);
      fileHeader.appendChild(relevance);
      
      const filePath = document.createElement('div');
      filePath.className = 'file-path';
      filePath.textContent = file.file_path;
      
      fileItem.appendChild(fileHeader);
      fileItem.appendChild(filePath);
      
      // Add GitHub link if we have repo info
      if (currentRepoId) {
        const githubLink = document.createElement('a');
        githubLink.className = 'github-link';
        githubLink.href = `https://github.com/${currentRepoId}/blob/main/${file.file_path}`;
        githubLink.target = '_blank';
        githubLink.textContent = 'View on GitHub';
        fileItem.appendChild(githubLink);
      }
      
      filesContent.appendChild(fileItem);
    });
    
    filesSidebar.appendChild(toggleBtn);
    filesSidebar.appendChild(filesContent);
    messageDiv.appendChild(filesSidebar);
  }

  chatContainer.appendChild(messageDiv);
  chatContainer.scrollTop = chatContainer.scrollHeight;
  
  // Apply syntax highlighting to any code blocks
  messageDiv.querySelectorAll('pre code').forEach((block) => {
    Prism.highlightElement(block);
  });
}

function updateStatusPanel(status, message, cacheStats = null) {
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

  let cacheHtml = '';
  if (cacheStats && (cacheStats.cache_hits > 0 || cacheStats.cache_misses > 0)) {
    cacheHtml = `
      <div class="status-item">
        <span class="status-label">🗄️ Cache:</span>
        <span class="status-value">${cacheStats.cache_hit_rate}% hit rate (${cacheStats.cache_hits} hits, ${cacheStats.cache_misses} misses)</span>
      </div>
    `;
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
    ${cacheHtml}
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

// Theme toggle functionality
function toggleTheme() {
  const currentTheme = document.documentElement.getAttribute('data-theme');
  const newTheme = currentTheme === 'light' ? 'dark' : 'light';
  
  setTheme(newTheme);
}

function setTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('theme', theme);
  
  // Update theme icon
  const themeIcon = document.querySelector('.theme-icon');
  if (themeIcon) {
    themeIcon.textContent = theme === 'light' ? '☀️' : '🌙';
  }
}

// Initialize theme on page load
function initTheme() {
  const savedTheme = localStorage.getItem('theme') || 'dark';
  setTheme(savedTheme);
}

// Initialize theme when DOM is loaded
document.addEventListener('DOMContentLoaded', initTheme);

console.log(`Repo-Trace frontend initialized. API Base: ${CONFIG.API_BASE}`);
