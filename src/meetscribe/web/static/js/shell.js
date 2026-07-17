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
            return window.location.pathname.startsWith('/sessions') ? 'sessions' : 'workflow';
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
