// MeetScribe Web UI - Alpine.js Application

function app() {
    return {
        // State
        session: null,
        currentStep: 1,
        globalSpeakers: [],

        // Upload state
        uploading: false,
        uploadProgress: 0,
        isDragging: false,

        // Extraction state
        extracting: false,
        extractionComplete: false,
        extractionStatus: '',
        extractionStep: 0,
        extractionTotal: 1,
        extractionLogs: [],

        // Enrollment state
        enrolling: false,
        enrollmentComplete: false,
        enrollmentStatus: '',
        enrollmentLogs: [],

        // Transcription state
        transcribing: false,
        transcriptionComplete: false,
        transcriptionStatus: '',
        transcriptionStep: 0,
        transcriptionTotal: 1,
        transcriptionProgress: 0,  // Progress within current step (0-100)
        transcriptionLogs: [],
        whisperModel: 'medium',
        language: 'ru',

        // Samples state
        playingSample: null,
        sortables: [],

        // Add speaker state
        addingSpeaker: false,
        newSpeakerName: '',
        speakerSuggestions: [],

        // Audio player state
        playbackRate: 1,
        sampleProgress: 0,
        sampleDuration: 0,
        sampleCurrentTime: 0,
        currentSampleInfo: null,

        steps: [
            { name: 'Upload' },
            { name: 'Configure' },
            { name: 'Extract' },
            { name: 'Samples' },
            { name: 'Enroll' },
            { name: 'Transcribe' }
        ],

        async init() {
            // Check URL for existing session
            const params = new URLSearchParams(window.location.search);
            const sessionId = params.get('session');
            const step = parseInt(params.get('step')) || 1;

            if (sessionId) {
                // Restore existing session
                this.session = { id: sessionId };
                try {
                    await this.loadSession();
                    // Restore state based on session status
                    const status = this.session?.status;
                    if (status === 'extracted' || status === 'enrolled' || status === 'transcribed') {
                        this.extractionComplete = true;
                    }
                    if (status === 'enrolled' || status === 'transcribed') {
                        this.enrollmentComplete = true;
                    }
                    if (status === 'transcribed' && this.session?.transcript) {
                        this.transcriptionComplete = true;
                    }
                    // Restore step (with validation)
                    if (step >= 1 && step <= 6) {
                        this.currentStep = step;
                        // Initialize sortables if on samples step
                        if (step === 4) {
                            this.$nextTick(() => this.initSortables());
                        }
                    }
                } catch (error) {
                    console.error('Failed to restore session:', error);
                    await this.createSession();
                }
            } else {
                await this.createSession();
            }
            await this.loadGlobalSpeakers();
        },

        updateUrl() {
            if (!this.session?.id) return;
            const url = new URL(window.location);
            url.searchParams.set('session', this.session.id);
            url.searchParams.set('step', this.currentStep);
            window.history.replaceState({}, '', url);
        },

        async createSession() {
            try {
                const response = await fetch('/api/session', { method: 'POST' });
                const data = await response.json();
                this.session = { id: data.session_id, tracks: [], speakers: [], samples: [] };
                this.updateUrl();
            } catch (error) {
                console.error('Failed to create session:', error);
            }
        },

        async loadSession() {
            if (!this.session?.id) return;
            try {
                const response = await fetch(`/api/session/${this.session.id}`);
                this.session = await response.json();
            } catch (error) {
                console.error('Failed to load session:', error);
            }
        },

        async loadGlobalSpeakers() {
            try {
                const response = await fetch('/api/speakers');
                this.globalSpeakers = await response.json();
            } catch (error) {
                console.error('Failed to load speakers:', error);
            }
        },

        canGoToStep(step) {
            if (step <= this.currentStep) return true;
            if (step === 2) return this.session?.tracks?.length > 0;

            // Steps 3-5 require diarization
            if (step >= 3 && step <= 5) {
                if (!this.needsDiarization()) return false;
                if (step === 3) return this.session?.tracks?.length > 0;
                if (step === 4) return this.extractionComplete;
                if (step === 5) return this.extractionComplete;
            }

            if (step === 6) return true;
            return false;
        },

        stopAllAudio() {
            // Stop track audio player
            const audioPlayer = this.$refs.audioPlayer;
            if (audioPlayer) {
                audioPlayer.pause();
                audioPlayer.currentTime = 0;
            }
            // Stop sample audio player
            const samplePlayer = this.$refs.samplePlayer;
            if (samplePlayer) {
                samplePlayer.pause();
                samplePlayer.currentTime = 0;
            }
            this.playingSample = null;
            this.currentSampleInfo = null;
            this.sampleProgress = 0;
            this.sampleCurrentTime = 0;
            this.sampleDuration = 0;
        },

        goToStep(step) {
            if (this.canGoToStep(step)) {
                this.stopAllAudio();
                this.currentStep = step;
                this.updateUrl();
                // Initialize sortables when entering samples step
                if (step === 4) {
                    this.$nextTick(() => this.initSortables());
                }
            }
        },

        needsDiarization() {
            // Check if any track needs diarization
            return this.session?.tracks?.some(t => t.diarize) ?? false;
        },

        nextStep() {
            if (this.currentStep < 6) {
                this.stopAllAudio();

                // From step 2 (Configure): skip samples steps if no diarization needed
                if (this.currentStep === 2 && !this.needsDiarization()) {
                    this.currentStep = 6;  // Go directly to Transcribe
                } else {
                    this.currentStep++;
                }

                this.updateUrl();
                // Initialize sortables when entering samples step
                if (this.currentStep === 4) {
                    this.$nextTick(() => this.initSortables());
                }
            }
        },

        prevStep() {
            if (this.currentStep > 1) {
                this.stopAllAudio();

                // From step 6 (Transcribe): skip samples steps if no diarization needed
                if (this.currentStep === 6 && !this.needsDiarization()) {
                    this.currentStep = 2;  // Go back to Configure
                } else {
                    this.currentStep--;
                }

                this.updateUrl();
            }
        },

        // Dropzone Methods
        handleDrop(event) {
            this.isDragging = false;
            const files = event.dataTransfer.files;
            if (files.length > 0) {
                this.uploadFiles(files);
            }
        },

        handleFiles(files) {
            if (files.length > 0) {
                this.uploadFiles(files);
            }
        },

        // Upload Methods
        async uploadFiles(files) {
            if (!this.session?.id || !files.length) return;

            this.uploading = true;
            this.uploadProgress = 0;

            const formData = new FormData();
            for (const file of files) {
                formData.append('files', file);
            }

            try {
                await new Promise((resolve, reject) => {
                    const xhr = new XMLHttpRequest();

                    xhr.upload.addEventListener('progress', (event) => {
                        if (event.lengthComputable) {
                            this.uploadProgress = Math.round((event.loaded / event.total) * 100);
                        }
                    });

                    xhr.addEventListener('load', () => {
                        if (xhr.status >= 200 && xhr.status < 300) {
                            resolve();
                        } else {
                            reject(new Error(xhr.responseText || 'Upload failed'));
                        }
                    });

                    xhr.addEventListener('error', () => {
                        reject(new Error('Network error'));
                    });

                    xhr.open('POST', `/api/session/${this.session.id}/tracks`);
                    xhr.send(formData);
                });

                await this.loadSession();
                this.uploadProgress = 100;
            } catch (error) {
                console.error('Upload failed:', error);
                alert('Upload failed: ' + error.message);
            } finally {
                this.uploading = false;
            }
        },

        async deleteTrack(trackNum) {
            if (!this.session?.id) return;

            try {
                await fetch(`/api/session/${this.session.id}/tracks/${trackNum}`, {
                    method: 'DELETE'
                });
                await this.loadSession();
            } catch (error) {
                console.error('Delete failed:', error);
            }
        },

        playTrack(trackNum) {
            const player = this.$refs.audioPlayer;
            if (player) {
                player.src = `/api/session/${this.session.id}/tracks/${trackNum}/audio`;
                player.play();
            }
        },

        // Config Methods
        async updateTrackConfig(trackNum, speakerName, diarize) {
            if (!this.session?.id) return;

            const params = new URLSearchParams();
            if (speakerName) params.set('speaker_name', speakerName);
            params.set('diarize', diarize);

            try {
                await fetch(`/api/session/${this.session.id}/tracks/${trackNum}?${params}`, {
                    method: 'PATCH'
                });
                await this.loadSession();
            } catch (error) {
                console.error('Update failed:', error);
            }
        },

        // Extraction Methods
        async startExtraction() {
            if (!this.session?.id) return;

            this.extracting = true;
            this.extractionComplete = false;
            this.extractionLogs = [];
            this.extractionStep = 0;
            this.extractionTotal = 1;

            try {
                // Start extraction
                await fetch(`/api/session/${this.session.id}/extract`, { method: 'POST' });

                // Connect to SSE stream
                const eventSource = new EventSource(`/api/session/${this.session.id}/extract/stream`);

                eventSource.onmessage = async (event) => {
                    const data = JSON.parse(event.data);

                    if (data.done) {
                        eventSource.close();
                        this.extracting = false;
                        this.extractionComplete = true;
                        await this.loadSession();
                        return;
                    }

                    if (data.error) {
                        eventSource.close();
                        this.extracting = false;
                        this.extractionLogs.push(`Error: ${data.error}`);
                        return;
                    }

                    if (data.step) {
                        this.extractionStep = data.step;
                        this.extractionTotal = data.total;
                    }
                    if (data.message) {
                        this.extractionStatus = data.message;
                        this.extractionLogs.push(data.message);
                    }
                };

                eventSource.onerror = () => {
                    eventSource.close();
                    this.extracting = false;
                };
            } catch (error) {
                console.error('Extraction failed:', error);
                this.extracting = false;
            }
        },

        // Samples Methods
        get unassignedSamples() {
            if (!this.session?.samples) return [];
            return this.session.samples.filter(s => !s.speaker_id && !s.is_known);
        },

        get hasAssignedSamples() {
            if (!this.session?.samples || !this.session?.speakers) return false;
            return this.session.speakers.some(speaker =>
                this.session.samples.some(sample => sample.speaker_id === speaker.id)
            );
        },

        getSpeakerSamples(speakerId) {
            if (!this.session?.samples) return [];
            return this.session.samples.filter(s => s.speaker_id === speakerId);
        },

        initSortables() {
            // Clean up existing sortables
            this.sortables.forEach(s => s.destroy());
            this.sortables = [];

            const containers = document.querySelectorAll('.speaker-bin');
            const sessionId = this.session?.id;
            const self = this;

            containers.forEach(container => {
                const sortable = new Sortable(container, {
                    group: 'samples',
                    animation: 150,
                    ghostClass: 'drag-ghost',
                    chosenClass: 'drag-chosen',
                    onEnd: async (evt) => {
                        const sampleId = evt.item.dataset.sampleId;
                        const newSpeakerId = evt.to.dataset.speakerId || null;

                        if (sessionId && sampleId) {
                            try {
                                await fetch(`/api/session/${sessionId}/samples/${sampleId}/move`, {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({ speaker_id: newSpeakerId })
                                });
                                await self.loadSession();
                                self.enrollmentComplete = false;
                            } catch (error) {
                                console.error('Move failed:', error);
                            }
                        }
                    }
                });
                this.sortables.push(sortable);
            });
        },

        showAddSpeakerInput() {
            this.addingSpeaker = true;
            this.newSpeakerName = '';
            this.speakerSuggestions = [...this.globalSpeakers];
            this.$nextTick(() => this.$refs.speakerInput?.focus());
        },

        cancelAddSpeaker() {
            this.addingSpeaker = false;
            this.newSpeakerName = '';
            this.speakerSuggestions = [];
        },

        filterSpeakers() {
            const query = this.newSpeakerName.toLowerCase().trim();
            if (!query) {
                this.speakerSuggestions = [...this.globalSpeakers];
            } else {
                this.speakerSuggestions = this.globalSpeakers.filter(
                    s => s.name.toLowerCase().includes(query)
                );
            }
        },

        selectSpeaker(name) {
            this.newSpeakerName = name;
            this.speakerSuggestions = [];
        },

        async confirmAddSpeaker() {
            const name = this.newSpeakerName.trim();
            if (!name || !this.session?.id) return;

            try {
                await fetch(`/api/session/${this.session.id}/speakers`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name })
                });
                await this.loadSession();
                this.$nextTick(() => this.initSortables());
                this.enrollmentComplete = false;
            } catch (error) {
                console.error('Add speaker failed:', error);
            }

            this.addingSpeaker = false;
            this.newSpeakerName = '';
            this.speakerSuggestions = [];
        },

        async renameSpeaker(speakerId, name) {
            if (!this.session?.id || !name) return;

            try {
                await fetch(`/api/session/${this.session.id}/speakers/${speakerId}`, {
                    method: 'PATCH',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name })
                });
                await this.loadSession();
            } catch (error) {
                console.error('Rename failed:', error);
            }
        },

        async deleteSpeakerBin(speakerId) {
            if (!confirm('Delete this speaker bin? Samples will be moved to unassigned.')) return;

            if (!this.session?.id) return;

            try {
                await fetch(`/api/session/${this.session.id}/speakers/${speakerId}`, {
                    method: 'DELETE'
                });
                await this.loadSession();
                this.$nextTick(() => this.initSortables());
                this.enrollmentComplete = false;
            } catch (error) {
                console.error('Delete failed:', error);
            }
        },

        async deleteSample(sampleId) {
            if (!confirm('Delete this sample?')) return;

            if (!this.session?.id) return;

            try {
                await fetch(`/api/session/${this.session.id}/samples/${sampleId}`, {
                    method: 'DELETE'
                });
                await this.loadSession();
                this.enrollmentComplete = false;
            } catch (error) {
                console.error('Delete failed:', error);
            }
        },

        playSample(sampleId) {
            if (!this.session?.id) return;

            const player = this.$refs.samplePlayer;
            if (!player) return;

            if (this.playingSample === sampleId) {
                player.pause();
                this.playingSample = null;
                this.currentSampleInfo = null;
            } else {
                player.src = `/api/session/${this.session.id}/samples/${sampleId}/audio`;
                player.playbackRate = this.playbackRate;
                player.play();
                this.playingSample = sampleId;
                // Find sample info for display
                const sample = this.session.samples?.find(s => s.id === sampleId);
                if (sample) {
                    this.currentSampleInfo = {
                        trackNum: sample.track_num,
                        clusterId: sample.cluster_id,
                        durationMs: sample.duration_ms
                    };
                }
            }
        },

        stopSample() {
            this.playingSample = null;
            this.currentSampleInfo = null;
            this.sampleProgress = 0;
            this.sampleCurrentTime = 0;
            this.sampleDuration = 0;
        },

        updateSampleProgress() {
            const player = this.$refs.samplePlayer;
            if (player && player.duration) {
                this.sampleCurrentTime = player.currentTime;
                this.sampleDuration = player.duration;
                this.sampleProgress = (player.currentTime / player.duration) * 100;
            }
        },

        setPlaybackSpeed(rate) {
            this.playbackRate = rate;
            const player = this.$refs.samplePlayer;
            if (player) {
                player.playbackRate = rate;
            }
        },

        seekSample(seconds) {
            const player = this.$refs.samplePlayer;
            if (player && player.duration) {
                player.currentTime = Math.max(0, Math.min(player.duration, player.currentTime + seconds));
            }
        },

        seekToPercent(event) {
            const player = this.$refs.samplePlayer;
            if (player && player.duration) {
                const rect = event.currentTarget.getBoundingClientRect();
                const percent = (event.clientX - rect.left) / rect.width;
                player.currentTime = player.duration * Math.max(0, Math.min(1, percent));
            }
        },

        formatTime(seconds) {
            if (!seconds || isNaN(seconds)) return '0.0s';
            return seconds.toFixed(1) + 's';
        },

        // Enrollment Methods
        get hasSpeakersToEnroll() {
            if (!this.session?.speakers?.length) return false;
            return this.session.speakers.some(s =>
                this.session.samples.some(sample => sample.speaker_id === s.id)
            );
        },

        getSpeakerSampleCount(speakerId) {
            if (!this.session?.samples) return 0;
            return this.session.samples.filter(s => s.speaker_id === speakerId).length;
        },

        async startEnrollment() {
            if (!this.session?.id) return;

            this.enrolling = true;
            this.enrollmentComplete = false;
            this.enrollmentLogs = [];

            try {
                await fetch(`/api/session/${this.session.id}/enroll`, { method: 'POST' });

                const eventSource = new EventSource(`/api/session/${this.session.id}/enroll/stream`);

                eventSource.onmessage = async (event) => {
                    const data = JSON.parse(event.data);

                    if (data.done) {
                        eventSource.close();
                        this.enrolling = false;
                        this.enrollmentComplete = true;
                        await this.loadGlobalSpeakers();
                        return;
                    }

                    if (data.error) {
                        eventSource.close();
                        this.enrolling = false;
                        this.enrollmentLogs.push(`Error: ${data.error}`);
                        return;
                    }

                    if (data.message) {
                        this.enrollmentStatus = data.message;
                        this.enrollmentLogs.push(data.message);
                    }
                };

                eventSource.onerror = () => {
                    eventSource.close();
                    this.enrolling = false;
                };
            } catch (error) {
                console.error('Enrollment failed:', error);
                this.enrolling = false;
            }
        },

        // Transcription Methods
        async startTranscription() {
            if (!this.session?.id) return;

            this.transcribing = true;
            this.transcriptionComplete = false;
            this.transcriptionLogs = [];
            this.transcriptionStep = 0;
            this.transcriptionTotal = 1;
            this.transcriptionProgress = 0;

            try {
                await fetch(`/api/session/${this.session.id}/transcribe`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: this.whisperModel,
                        language: this.language
                    })
                });

                const eventSource = new EventSource(`/api/session/${this.session.id}/transcribe/stream`);
                let lastStep = 0;

                eventSource.onmessage = async (event) => {
                    const data = JSON.parse(event.data);

                    if (data.done) {
                        eventSource.close();
                        this.transcribing = false;
                        this.transcriptionComplete = true;
                        await this.loadSession();
                        return;
                    }

                    if (data.error) {
                        eventSource.close();
                        this.transcribing = false;
                        this.transcriptionLogs.push(`Error: ${data.error}`);
                        return;
                    }

                    if (data.step) {
                        // Reset progress when step changes
                        if (data.step !== lastStep) {
                            this.transcriptionProgress = 0;
                            lastStep = data.step;
                        }
                        this.transcriptionStep = data.step;
                        this.transcriptionTotal = data.total;
                    }
                    // Update sub-step progress (0-100 within current step)
                    if (data.progress !== undefined) {
                        this.transcriptionProgress = data.progress;
                    }
                    if (data.message) {
                        this.transcriptionStatus = data.message;
                        // Only log non-progress messages (avoid flooding logs)
                        if (data.progress === undefined || data.progress === 0 || data.progress === 100) {
                            this.transcriptionLogs.push(data.message);
                        }
                    }
                };

                eventSource.onerror = () => {
                    eventSource.close();
                    this.transcribing = false;
                };
            } catch (error) {
                console.error('Transcription failed:', error);
                this.transcribing = false;
            }
        },

        formatMarkdown(text) {
            if (!text) return '';
            // Simple markdown to HTML conversion
            return text
                .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
                .replace(/\n\n/g, '<br><br>');
        },

        copyTranscript() {
            if (this.session?.transcript) {
                navigator.clipboard.writeText(this.session.transcript);
                alert('Transcript copied to clipboard!');
            }
        },

        downloadTranscript() {
            if (this.session?.transcript) {
                const blob = new Blob([this.session.transcript], { type: 'text/markdown' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'transcript.md';
                a.click();
                URL.revokeObjectURL(url);
            }
        },

        async startNewSession() {
            this.session = null;
            this.currentStep = 1;
            this.extractionComplete = false;
            this.extractionLogs = [];
            this.enrollmentComplete = false;
            this.enrollmentLogs = [];
            this.transcriptionComplete = false;
            this.transcriptionLogs = [];
            // Clear URL params before creating new session
            window.history.replaceState({}, '', window.location.pathname);
            await this.createSession();
        }
    };
}
