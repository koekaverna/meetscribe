// MeetScribe Web UI - shell: auth fetch, page routing, page mounting.
// Pages (workflowPage, sessionsPage) are mounted via x-if in index.html;
// bumping workflowKey re-mounts the workflow so it re-reads ?session=&step=.

// Fetch wrapper that redirects to login on 401
async function authFetch(url, options = {}) {
    const response = await fetch(url, options);
    if (response.status === 401) {
        window.location.href = '/login';
        throw new Error('Not authenticated');
    }
    return response;
}

// Read a cookie value by name (null if absent).
function getCookie(name) {
    const match = document.cookie.match(new RegExp('(?:^|; )' + name + '=([^;]*)'));
    return match ? decodeURIComponent(match[1]) : null;
}

// Global translation helper for JS/Alpine. Reads the catalog injected in <head>
// (window.__I18N__). Missing key -> returns the key. Interpolates {name} tokens
// from `params` (named braces, same as the Python-side t()).
window.t = function (key, params) {
    const catalog = window.__I18N__ || {};
    let text = Object.prototype.hasOwnProperty.call(catalog, key) ? catalog[key] : key;
    if (params) {
        text = text.replace(/\{(\w+)\}/g, (m, name) =>
            Object.prototype.hasOwnProperty.call(params, name) ? params[name] : m
        );
    }
    return text;
};

// Persist the UI language (cookie + users.lang) then reload to re-render.
window.setLang = async function (lang) {
    if (lang === window.__LANG__) return;
    const form = new URLSearchParams();
    form.set('lang', lang);
    form.set('csrf_token', getCookie('meetscribe_csrf') || '');
    try {
        await fetch('/api/lang', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: form.toString(),
        });
    } finally {
        window.location.reload();
    }
};

document.addEventListener('alpine:init', () => {
    Alpine.data('shell', () => ({
        // Named distinctively: shell methods are called from page-component scopes,
        // where `this` is the merged scope chain — a page property with the same
        // name (e.g. sessionsPage's pagination `page`) would shadow the write.
        activePage: null,
        workflowKey: 0,

        init() {
            this.activePage = this._pageFromPath();
            window.addEventListener('popstate', () => {
                this.activePage = this._pageFromPath();
                this.workflowKey++;
            });
        },

        _pageFromPath() {
            const path = window.location.pathname;
            if (path.startsWith('/sessions')) return 'sessions';
            if (path.startsWith('/speakers')) return 'speakers';
            if (path.startsWith('/admin')) return 'admin';
            return 'workflow';
        },

        navigate(path) {
            window.history.pushState({}, '', path);
            this.activePage = this._pageFromPath();
        },

        newWorkflowSession() {
            window.history.replaceState({}, '', '/');
            this.activePage = 'workflow';
            this.workflowKey++;
        },

        openWorkflow(sessionId, step) {
            const url = `/?session=${sessionId}${step ? `&step=${step}` : ''}`;
            window.history.pushState({}, '', url);
            this.activePage = 'workflow';
            this.workflowKey++;
        },
    }));
});
