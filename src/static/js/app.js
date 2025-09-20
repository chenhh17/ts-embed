// Global JavaScript functionality for Research Paper Explorer

// Utility functions
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
        year: 'numeric', 
        month: 'short', 
        day: 'numeric' 
    });
}

function truncateText(text, maxLength) {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

function highlightSearchTerms(text, searchTerms) {
    if (!searchTerms || searchTerms.length === 0) return text;
    
    let highlightedText = text;
    searchTerms.forEach(term => {
        const regex = new RegExp(`(${term})`, 'gi');
        highlightedText = highlightedText.replace(regex, '<mark>$1</mark>');
    });
    
    return highlightedText;
}

// Navigation helpers
function setActiveNavLink() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
    });
}

// Toast notifications
function showToast(message, type = 'info') {
    const toastContainer = document.getElementById('toast-container') || createToastContainer();
    
    const toastId = 'toast-' + Date.now();
    const toastHtml = `
        <div id="${toastId}" class="toast align-items-center text-white bg-${type} border-0" role="alert">
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" 
                        data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;
    
    toastContainer.insertAdjacentHTML('beforeend', toastHtml);
    
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement);
    toast.show();
    
    // Remove toast element after it's hidden
    toastElement.addEventListener('hidden.bs.toast', () => {
        toastElement.remove();
    });
}

function createToastContainer() {
    const containerHtml = `
        <div id="toast-container" class="toast-container position-fixed bottom-0 end-0 p-3">
        </div>
    `;
    document.body.insertAdjacentHTML('beforeend', containerHtml);
    return document.getElementById('toast-container');
}

// Copy to clipboard functionality
function copyToClipboard(text, successMessage = 'Copied to clipboard!') {
    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(() => {
            showToast(successMessage, 'success');
        }).catch(err => {
            console.error('Failed to copy: ', err);
            fallbackCopyToClipboard(text, successMessage);
        });
    } else {
        fallbackCopyToClipboard(text, successMessage);
    }
}

function fallbackCopyToClipboard(text, successMessage) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.left = '-999999px';
    textArea.style.top = '-999999px';
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    
    try {
        document.execCommand('copy');
        showToast(successMessage, 'success');
    } catch (err) {
        console.error('Fallback copy failed: ', err);
        showToast('Failed to copy to clipboard', 'danger');
    }
    
    document.body.removeChild(textArea);
}

// Local storage helpers
function saveToLocalStorage(key, data) {
    try {
        localStorage.setItem(key, JSON.stringify(data));
    } catch (error) {
        console.error('Failed to save to localStorage:', error);
    }
}

function loadFromLocalStorage(key, defaultValue = null) {
    try {
        const item = localStorage.getItem(key);
        return item ? JSON.parse(item) : defaultValue;
    } catch (error) {
        console.error('Failed to load from localStorage:', error);
        return defaultValue;
    }
}

// Search history management
function saveSearchHistory(query) {
    const history = loadFromLocalStorage('searchHistory', []);
    
    // Remove if already exists
    const filteredHistory = history.filter(item => item.query !== query);
    
    // Add to beginning
    filteredHistory.unshift({
        query: query,
        timestamp: new Date().toISOString()
    });
    
    // Keep only last 10 searches
    const limitedHistory = filteredHistory.slice(0, 10);
    
    saveToLocalStorage('searchHistory', limitedHistory);
}

function getSearchHistory() {
    return loadFromLocalStorage('searchHistory', []);
}

function clearSearchHistory() {
    localStorage.removeItem('searchHistory');
    showToast('Search history cleared', 'info');
}

// Debounce function for search
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Error handling
function handleApiError(error, defaultMessage = 'An error occurred') {
    console.error('API Error:', error);
    
    let message = defaultMessage;
    if (error.response) {
        // Server responded with error status
        message = error.response.data?.message || `Server error: ${error.response.status}`;
    } else if (error.request) {
        // Request made but no response
        message = 'Network error. Please check your connection.';
    }
    
    showToast(message, 'danger');
}

// Loading state management
function showLoading(elementId, message = 'Loading...') {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `
            <div class="text-center loading">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-2 text-muted">${message}</p>
            </div>
        `;
    }
}

function hideLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = '';
    }
}

// URL helpers
function updateURLParams(params) {
    const url = new URL(window.location);
    Object.keys(params).forEach(key => {
        if (params[key] !== null && params[key] !== undefined && params[key] !== '') {
            url.searchParams.set(key, params[key]);
        } else {
            url.searchParams.delete(key);
        }
    });
    window.history.replaceState({}, '', url);
}

function getURLParams() {
    const params = {};
    const urlParams = new URLSearchParams(window.location.search);
    for (const [key, value] of urlParams) {
        params[key] = value;
    }
    return params;
}

// Initialize common functionality
document.addEventListener('DOMContentLoaded', function() {
    // Set active navigation link
    setActiveNavLink();
    
    // Add copy DOI functionality to all DOI elements
    document.addEventListener('click', function(e) {
        if (e.target.matches('[data-copy-doi]')) {
            const doi = e.target.getAttribute('data-copy-doi');
            copyToClipboard(doi, 'DOI copied to clipboard!');
        }
    });
    
    // Handle URL parameters on page load
    const urlParams = getURLParams();
    if (urlParams.query && document.getElementById('search-query')) {
        document.getElementById('search-query').value = urlParams.query;
        // Trigger search if on search page
        if (window.location.pathname === '/search') {
            setTimeout(() => {
                if (typeof performSearch === 'function') {
                    performSearch();
                }
            }, 100);
        }
    }
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl+K or Cmd+K to focus search
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            const searchInput = document.getElementById('search-query');
            if (searchInput) {
                searchInput.focus();
                searchInput.select();
            } else if (window.location.pathname !== '/search') {
                window.location.href = '/search';
            }
        }
        
        // Escape to clear search
        if (e.key === 'Escape') {
            const searchInput = document.getElementById('search-query');
            if (searchInput && document.activeElement === searchInput) {
                searchInput.value = '';
                searchInput.blur();
            }
        }
    });
});

// Export functions for use in other scripts
window.PaperExplorer = {
    formatDate,
    truncateText,
    highlightSearchTerms,
    showToast,
    copyToClipboard,
    saveToLocalStorage,
    loadFromLocalStorage,
    saveSearchHistory,
    getSearchHistory,
    clearSearchHistory,
    debounce,
    handleApiError,
    showLoading,
    hideLoading,
    updateURLParams,
    getURLParams
};
