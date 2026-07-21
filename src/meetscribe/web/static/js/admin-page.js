// MeetScribe Web UI - admin panel page.
// Users/teams management, Speaches server status, disk usage, recent errors.

document.addEventListener('alpine:init', () => {
    Alpine.data('adminPage', (isSuperadmin = false, currentUser = '') => ({
        isSuperadmin,
        currentUser,
        users: [],
        teams: [],
        servers: [],
        disk: null,
        errorLines: [],
        errorFile: null,
        statusLoading: false,
        actionError: null,
        newUser: { username: '', password: '', team: 'default', isAdmin: false },
        newTeam: { name: '', description: '' },

        init() {
            this.loadUsers();
            // Teams, server status, disk and errors are superadmin-only APIs
            if (this.isSuperadmin) {
                this.loadTeams();
                this.loadStatus();
                this.loadDisk();
                this.loadErrors();
            }
        },

        async _get(url) {
            return (await this._send(url)).json();
        },

        // Mutations surface the API's error detail in the banner
        async _send(url, options = {}) {
            const response = await authFetch(url, options);
            if (!response.ok) {
                const data = await response.json().catch(() => ({}));
                throw new Error(data.detail || `HTTP ${response.status}`);
            }
            return response;
        },

        async loadUsers() {
            try {
                this.users = await this._get('/api/admin/users');
            } catch (error) {
                console.error('Failed to load users:', error);
            }
        },

        async loadTeams() {
            try {
                this.teams = await this._get('/api/admin/teams');
            } catch (error) {
                console.error('Failed to load teams:', error);
            }
        },

        async loadStatus() {
            this.statusLoading = true;
            try {
                this.servers = await this._get('/api/admin/status');
            } catch (error) {
                console.error('Failed to load server status:', error);
            } finally {
                this.statusLoading = false;
            }
        },

        async loadDisk() {
            try {
                this.disk = await this._get('/api/admin/disk');
            } catch (error) {
                console.error('Failed to load disk usage:', error);
            }
        },

        async loadErrors() {
            try {
                const data = await this._get('/api/admin/errors');
                this.errorLines = data.lines;
                this.errorFile = data.file;
            } catch (error) {
                console.error('Failed to load error log:', error);
            }
        },

        async createUser() {
            this.actionError = null;
            try {
                await this._send('/api/admin/users', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        username: this.newUser.username,
                        password: this.newUser.password,
                        // Team admins omit the team: the server uses their own
                        ...(this.isSuperadmin && { team_name: this.newUser.team }),
                        is_admin: this.newUser.isAdmin,
                    }),
                });
                this.newUser = { username: '', password: '', team: 'default', isAdmin: false };
                await this._reloadLists();
            } catch (error) {
                this.actionError = error.message;
            }
        },

        async deleteUser(u) {
            const msg = `Delete user "${u.username}"?\n\n`
                + 'Their sessions will remain but lose their creator. This cannot be undone.';
            if (!confirm(msg)) return;
            this.actionError = null;
            try {
                await this._send(`/api/admin/users/${encodeURIComponent(u.username)}`, { method: 'DELETE' });
                await this._reloadLists();
            } catch (error) {
                this.actionError = error.message;
            }
        },

        async toggleAdmin(u) {
            const msg = u.is_admin
                ? `Revoke admin from "${u.username}"?`
                : `Make "${u.username}" an admin of team "${u.team_name}"?`;
            if (!confirm(msg)) return;
            this.actionError = null;
            try {
                await this._send(`/api/admin/users/${encodeURIComponent(u.username)}`, {
                    method: 'PATCH',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ is_admin: !u.is_admin }),
                });
                await this.loadUsers();
            } catch (error) {
                this.actionError = error.message;
            }
        },

        async resetPassword(u) {
            const self = u.username === this.currentUser;
            const password = prompt(self
                ? 'New password for your own account (min 8 chars). You will be logged out:'
                : `New password for "${u.username}" (min 8 chars):`);
            if (password === null) return;
            this.actionError = null;
            try {
                await this._send(`/api/admin/users/${encodeURIComponent(u.username)}/password`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ password }),
                });
                if (self) {
                    // The reset invalidated our own session too
                    window.location.href = '/login';
                    return;
                }
                alert(`Password for "${u.username}" changed. Their sessions were logged out.`);
            } catch (error) {
                this.actionError = error.message;
            }
        },

        // Users always; teams only for superadmins (counts change with users)
        async _reloadLists() {
            const jobs = [this.loadUsers()];
            if (this.isSuperadmin) jobs.push(this.loadTeams());
            await Promise.all(jobs);
        },

        async createTeam() {
            this.actionError = null;
            try {
                await this._send('/api/admin/teams', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        name: this.newTeam.name,
                        description: this.newTeam.description || null,
                    }),
                });
                this.newTeam = { name: '', description: '' };
                await this.loadTeams();
            } catch (error) {
                this.actionError = error.message;
            }
        },

        async deleteTeam(t) {
            // Voiceprints cascade on team delete; users/sessions block it server-side
            const msg = `Delete team "${t.name}"?\n\n`
                + `This PERMANENTLY deletes all its voiceprints (${t.voiceprint_count}) `
                + 'and enrolled speaker samples. This cannot be undone.\n\n'
                + 'Teams that still have users or sessions cannot be deleted.';
            if (!confirm(msg)) return;
            this.actionError = null;
            try {
                await this._send(`/api/admin/teams/${encodeURIComponent(t.name)}`, { method: 'DELETE' });
                await this.loadTeams();
            } catch (error) {
                this.actionError = error.message;
            }
        },

        formatBytes(n) {
            if (n == null) return '—';
            const units = ['B', 'KB', 'MB', 'GB', 'TB'];
            let i = 0;
            while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
            return `${i === 0 ? n : n.toFixed(1)} ${units[i]}`;
        },

        formatDate(t) {
            // created_at is UTC text from SQLite datetime('now')
            return new Date(t.replace(' ', 'T') + 'Z').toLocaleString();
        },
    }));
});
