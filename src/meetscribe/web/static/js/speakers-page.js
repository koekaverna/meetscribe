// MeetScribe Web UI - enrolled speakers dashboard.
// Curate voiceprints: play/delete samples, rename/delete speakers.

document.addEventListener('alpine:init', () => {
    Alpine.data('speakersPage', () => ({
        speakers: [],
        speakersLoading: false,
        loadError: false,
        actionError: '',
        openSpeaker: null,   // name of the speaker whose samples are expanded
        samples: [],
        samplesLoading: false,
        samplesRequestId: 0, // discards stale responses after switching speakers
        playbackRate: 1,     // shared across all sample players
        volume: 1,           // volume/mute are shared too
        muted: false,
        _syncingVolume: false,
        renameTarget: null,
        renameValue: '',

        init() {
            this.load();
        },

        async load() {
            this.speakersLoading = true;
            try {
                const response = await authFetch('/api/speakers');
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                this.speakers = await response.json();
                this.loadError = false;
            } catch (error) {
                console.error('Failed to load speakers:', error);
                this.loadError = true;
            } finally {
                this.speakersLoading = false;
            }
        },

        async toggleSamples(name) {
            if (this.openSpeaker === name) {
                this.samplesRequestId++;
                this.openSpeaker = null;
                this.samples = [];
                this.samplesLoading = false;
                return;
            }
            this.openSpeaker = name;
            await this.loadSamples(name);
        },

        async loadSamples(name) {
            // An in-flight response for another speaker must not render (or be
            // deleted) under this one, so anything but the latest request is dropped
            const requestId = ++this.samplesRequestId;
            this.samplesLoading = true;
            this.samples = [];
            try {
                const response = await authFetch(
                    `/api/speakers/${encodeURIComponent(name)}/samples`);
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                const samples = await response.json();
                if (requestId !== this.samplesRequestId) return;
                this.samples = samples;
            } catch (error) {
                if (requestId !== this.samplesRequestId) return;
                console.error('Failed to load samples:', error);
                this.actionError = 'Failed to load samples.';
            } finally {
                if (requestId === this.samplesRequestId) this.samplesLoading = false;
            }
        },

        sampleUrl(filename) {
            return `/api/speakers/${encodeURIComponent(this.openSpeaker)}`
                + `/samples/${encodeURIComponent(filename)}/audio`;
        },

        // Only one sample plays at a time; shared settings are re-applied on play
        // because re-rendered <audio> elements start with browser defaults
        pauseOthers(current) {
            current.playbackRate = this.playbackRate;
            current.volume = this.volume;
            current.muted = this.muted;
            this.$root.querySelectorAll('audio').forEach(a => {
                if (a !== current) a.pause();
            });
        },

        setPlaybackSpeed(rate) {
            this.playbackRate = rate;
            this.$root.querySelectorAll('audio').forEach(a => { a.playbackRate = rate; });
        },

        // Volume/mute changed on one player propagates to the rest. Identical
        // assignments don't re-fire volumechange, so this settles in one pass.
        syncVolume(el) {
            if (this._syncingVolume) return;
            this.volume = el.volume;
            this.muted = el.muted;
            this._syncingVolume = true;
            this.$root.querySelectorAll('audio').forEach(a => {
                if (a !== el) { a.volume = this.volume; a.muted = this.muted; }
            });
            this._syncingVolume = false;
        },

        async removeSample(filename) {
            // Snapshot before awaiting — openSpeaker may change mid-request
            const speaker = this.openSpeaker;
            if (!confirm(`Delete sample "${filename}"? The voiceprint will be recomputed from the remaining samples.`)) return;
            this.actionError = '';
            try {
                const response = await authFetch(
                    `/api/speakers/${encodeURIComponent(speaker)}/samples/${encodeURIComponent(filename)}`,
                    { method: 'DELETE' });
                if (!response.ok) {
                    const data = await response.json().catch(() => ({}));
                    throw new Error(data.detail || `HTTP ${response.status}`);
                }
                const reloads = [this.load()];
                if (this.openSpeaker === speaker) reloads.push(this.loadSamples(speaker));
                await Promise.all(reloads);
            } catch (error) {
                console.error('Failed to delete sample:', error);
                this.actionError = String(error.message || error);
            }
        },

        async removeSpeaker(name) {
            if (!confirm(`Delete speaker "${name}" with all enrolled samples? This cannot be undone.`)) return;
            this.actionError = '';
            try {
                const response = await authFetch(
                    `/api/speakers/${encodeURIComponent(name)}`, { method: 'DELETE' });
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                if (this.openSpeaker === name) {
                    this.openSpeaker = null;
                    this.samples = [];
                }
                await this.load();
            } catch (error) {
                console.error('Failed to delete speaker:', error);
                this.actionError = 'Failed to delete speaker.';
            }
        },

        startRename(name) {
            this.renameTarget = name;
            this.renameValue = name;
            this.actionError = '';
        },

        async submitRename() {
            const newName = this.renameValue.trim();
            if (!newName || newName === this.renameTarget) {
                this.renameTarget = null;
                return;
            }
            try {
                const response = await authFetch(
                    `/api/speakers/${encodeURIComponent(this.renameTarget)}`, {
                        method: 'PATCH',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ name: newName }),
                    });
                if (!response.ok) {
                    const data = await response.json().catch(() => ({}));
                    throw new Error(data.detail || `HTTP ${response.status}`);
                }
                if (this.openSpeaker === this.renameTarget) this.openSpeaker = newName;
                this.renameTarget = null;
                await this.load();
            } catch (error) {
                console.error('Failed to rename speaker:', error);
                this.actionError = String(error.message || error);
            }
        },

        quality(s) {
            // Tiers follow enroll targets: samples are 4-12s each, up to 8 kept per speaker
            if (s.sample_count >= 3 && s.total_duration_ms >= 20000) return 'good';
            if (s.sample_count >= 2 && s.total_duration_ms >= 10000) return 'fair';
            return 'low';
        },

        qualityBadge(s) {
            return {
                good: 'bg-green-100 text-green-700',
                fair: 'bg-amber-100 text-amber-700',
                low: 'bg-red-100 text-red-700',
            }[this.quality(s)];
        },

        qualityStats(s) {
            const samples = s.sample_count === 1 ? 'sample' : 'samples';
            return `${s.sample_count} ${samples} · ${this.formatDuration(s.total_duration_ms)}`;
        },

        formatDuration(ms) {
            const s = Math.round(ms / 1000);
            return s >= 60 ? `${Math.floor(s / 60)}m ${s % 60}s` : `${s}s`;
        },

        formatDate(t) {
            // created_at is UTC text from SQLite datetime('now')
            return new Date(t.replace(' ', 'T') + 'Z').toLocaleString();
        },
    }));
});
