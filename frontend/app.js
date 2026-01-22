// Configuration
const CONFIG = {
    API_BASE: 'http://localhost:8000'
};

// State
let currentRepoId = null;
let statusPollInterval = null;
let isRepoReady = false;

// DOM Elements
const repoUrlInput = document.getElementById('repoUrl');
const branchInput = document.getElementById('branchInput');
const ingestBtn = document.getElementById('ingestBtn');
const statusPanel = document.getElementById('statusPanel');
const chatMessages = document.getElementById('chatMessages');
const questionInput = document.getElementById('questionInput');
const askBtn = document.getElementById('askBtn');

// Event Listeners
ingestBtn.addEventListener('click', handleIngest);
askBtn.addEventListener('click', handleAsk);
questionInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && e.ctrlKey && !askBtn.disabled) {
        handleAsk();
    }
});

/**
 * Extract repo_id from GitHub URL
 */
function extractRepoId(url) {
    try {
        const urlObj = new URL(url);
        const pathname = urlObj.pathname;
        const parts = pathname.split('/').filter(p => p);
        
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

/**
 * Handle repository ingestion
 */
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
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                repo_url: url,
                branch: branch
            })
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ message: response.statusText }));
            throw new Error(error.message || `HTTP ${response.status}`);
        }

        const data = await response.json();
        updateStatusPanel('processing', `Ingestion started. Repo ID: ${repoId}`);
        
        // Start polling status
        clearInterval(statusPollInterval);
        statusPollInterval = setInterval(() => pollStatus(repoId), 2000);
        
        // Initial poll
        pollStatus(repoId);
    } catch (error) {
        console.error('Ingest error:', error);
        updateStatusPanel('failed', `Error: ${error.message}`);
        resetIngestForm();
    }
}

/**
 * Poll ingestion status
 */
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
            updateStatusPanel('ready', 'Repository ready for querying');
            questionInput.disabled = false;
            askBtn.disabled = false;
            chatMessages.innerHTML = '<div class="empty-state">Ask a question about the repository.</div>';
        } else if (data.status === 'failed') {
            clearInterval(statusPollInterval);
            updateStatusPanel('failed', `Failed: ${data.error || 'Unknown error'}`);
            resetIngestForm();
        } else {
            // Still processing
            updateStatusPanel('processing', `${data.status || 'Processing'}${data.progress ? ` (${data.progress})` : ''}`);
        }
    } catch (error) {
        console.error('Status poll error:', error);
        updateStatusPanel('failed', 'Status check failed');
    }
}

/**
 * Handle question submission
 */
async function handleAsk() {
    const question = questionInput.value.trim();
    
    if (!question || !isRepoReady) {
        return;
    }

    // Add user message to chat
    addChatMessage('user', question);
    questionInput.value = '';
    questionInput.focus();
    
    // Disable input while waiting
    askBtn.disabled = true;
    questionInput.disabled = true;
    
    try {
        const response = await fetch(`${CONFIG.API_BASE}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                repo_id: currentRepoId,
                question: question
            })
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ message: response.statusText }));
            throw new Error(error.message || `HTTP ${response.status}`);
        }

        const data = await response.json();
        
        // Add assistant response
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

/**
 * Add message to chat
 */
function addChatMessage(role, content, citations = null) {
    const chatContainer = chatMessages;
    
    // Clear empty state if present
    const emptyState = chatContainer.querySelector('.empty-state');
    if (emptyState) {
        emptyState.remove();
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.textContent = content;
    messageDiv.appendChild(bubble);

    if (citations && citations.length > 0) {
        const citationsDiv = document.createElement('div');
        citationsDiv.className = 'citations';
        citationsDiv.innerHTML = '<strong>Sources:</strong>';
        
        citations.forEach(citation => {
            const link = document.createElement('a');
            link.className = 'citation-link';
            link.href = citation.url || '#';
            link.target = '_blank';
            link.textContent = citation.file || citation.path || citation.url || 'Source';
            citationsDiv.appendChild(link);
        });

        messageDiv.appendChild(citationsDiv);
    }

    chatContainer.appendChild(messageDiv);
    
    // Auto-scroll to bottom
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

/**
 * Update status panel
 */
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

/**
 * Show error message in status panel
 */
function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;
    
    const parent = statusPanel.parentElement;
    parent.insertBefore(errorDiv, parent.firstChild);
    
    setTimeout(() => errorDiv.remove(), 5000);
}

/**
 * Reset ingest form
 */
function resetIngestForm() {
    ingestBtn.disabled = false;
    repoUrlInput.disabled = false;
    branchInput.disabled = false;
    currentRepoId = null;
    isRepoReady = false;
}

// Initialize
console.log(`Repo-Trace frontend initialized. API Base: ${CONFIG.API_BASE}`);
