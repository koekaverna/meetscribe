// MeetScribe Web UI - session archive page.
// Opens a session in the workflow at the step matching its status.

const STATUS_STEP = {
    created: 1,
    uploaded: 2,
    // configured -> 2, not 3: only step 2's nextStep() knows whether diarization is needed
    configured: 2,
    extracted: 4,
    enrolled: 6,
    transcribed: 6,
};

document.addEventListener('alpine:init', () => {
    Alpine.data('sessionsPage', () => ({
        sessions: [],
        total: 0,
        page: 1,
        perPage: 20,
        sort: 'date',
        order: 'desc',
        mine: false,
        loading: false,
        loadError: false,
        selected: [],

        init() {
            this.load();
        },

        get totalPages() {
            return Math.max(1, Math.ceil(this.total / this.perPage));
        },

        async load() {
            this.loading = true;
            this.selected = [];
            try {
                const params = new URLSearchParams({
                    page: this.page,
                    per_page: this.perPage,
                    sort: this.sort,
                    order: this.order,
                    mine: this.mine,
                });
                const response = await authFetch(`/api/session?${params}`);
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                const data = await response.json();
                this.sessions = data.sessions;
                this.total = data.total;
                this.loadError = false;
            } catch (error) {
                console.error('Failed to load sessions:', error);
                this.loadError = true;
            } finally {
                this.loading = false;
            }
        },

        setSort(field) {
            if (this.sort === field) {
                this.order = this.order === 'desc' ? 'asc' : 'desc';
            } else {
                this.sort = field;
                this.order = 'desc';
            }
            this.page = 1;
            this.load();
        },

        toggleMine(mine) {
            if (this.mine === mine) return;
            this.mine = mine;
            this.page = 1;
            this.load();
        },

        prevPage() {
            if (this.page > 1) {
                this.page--;
                this.load();
            }
        },

        nextPage() {
            if (this.page < this.totalPages) {
                this.page++;
                this.load();
            }
        },

        toggleSelected(id) {
            this.selected = this.selected.includes(id)
                ? this.selected.filter(x => x !== id)
                : [...this.selected, id];
        },

        openSession(s) {
            // While selecting, a row click toggles the checkbox instead of navigating —
            // a stray click would otherwise throw away the whole selection.
            if (this.selected.length > 0) {
                this.toggleSelected(s.id);
                return;
            }
            // openWorkflow lives on the shell — resolves up the Alpine scope chain
            this.openWorkflow(s.id, STATUS_STEP[s.status] || 1);
        },

        async removeSession(id) {
            if (!confirm('Delete this session and all its files? This cannot be undone.')) return;
            try {
                await authFetch(`/api/session/${id}`, { method: 'DELETE' });
                // Don't strand the user on a page that just became empty
                if (this.sessions.length === 1 && this.page > 1) this.page--;
                await this.load();
            } catch (error) {
                console.error('Failed to delete session:', error);
            }
        },

        get allSelected() {
            return this.sessions.length > 0 && this.selected.length === this.sessions.length;
        },

        toggleSelectAll() {
            this.selected = this.allSelected ? [] : this.sessions.map(s => s.id);
        },

        async removeSelected() {
            const count = this.selected.length;
            const plural = count === 1 ? 'session' : 'sessions';
            if (!confirm(`Delete ${count} ${plural} with all their files? This cannot be undone.`)) return;
            try {
                const response = await authFetch('/api/session/bulk-delete', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ids: this.selected }),
                });
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                if (count === this.sessions.length && this.page > 1) this.page--;
                await this.load();
            } catch (error) {
                console.error('Bulk delete failed:', error);
            }
        },

        statusBadge(status) {
            return {
                created: 'bg-gray-100 text-gray-700',
                uploaded: 'bg-blue-100 text-blue-700',
                configured: 'bg-indigo-100 text-indigo-700',
                extracted: 'bg-amber-100 text-amber-700',
                enrolled: 'bg-purple-100 text-purple-700',
                transcribed: 'bg-green-100 text-green-700',
            }[status] || 'bg-gray-100 text-gray-700';
        },

        formatDate(t) {
            // created_at is UTC text from SQLite datetime('now')
            return new Date(t.replace(' ', 'T') + 'Z').toLocaleString();
        },

        formatDuration(ms) {
            if (ms == null) return '—';
            const m = Math.floor(ms / 60000);
            const s = Math.floor((ms % 60000) / 1000);
            return m >= 60 ? `${Math.floor(m / 60)}h ${m % 60}m` : `${m}m ${s}s`;
        },

        speakerLabel(list) {
            if (!list.length) return '—';
            return list.slice(0, 4).join(', ') + (list.length > 4 ? ` +${list.length - 4}` : '');
        },

        formatPreview(p) {
            return p ? p.replace(/[*#_]/g, '') : '';
        },
    }));
});
