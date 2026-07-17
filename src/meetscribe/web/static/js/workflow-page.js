// MeetScribe Web UI - workflow page component (6-step transcription flow).
// Mounted by the shell via x-if + keyed x-for; reads ?session=&step= from the URL.

// Full class names (not fragments) so the Tailwind CDN compiler picks them up.
const SPEAKER_COLORS = [
    'text-blue-700',
    'text-emerald-700',
    'text-purple-700',
    'text-rose-700',
    'text-amber-700',
    'text-cyan-700',
    'text-indigo-700',
    'text-orange-700',
    'text-teal-700',
    'text-fuchsia-700',
];

document.addEventListener('alpine:init', () => {
    Alpine.data('workflowPage', () => ({
        // State
        session: null,
        sessionNotFound: false,
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

        // UI feedback
        copied: false,

        // Audio player state (samples)
        playbackRate: 1,
        sampleProgress: 0,
        sampleDuration: 0,
        sampleCurrentTime: 0,
        currentSampleInfo: null,

        // Transcript editing state
        editingSegmentId: null,
        editSegmentText: '',
        editSegmentSpeaker: '',
        editSegmentNewName: '',
        editSegmentStart: '',
        editSegmentEnd: '',
        segmentBusy: false,
        insertAfterId: null,
        insertOpen: false,
        insertText: '',
        insertSpeaker: '',
        insertNewName: '',

        // Transcript player state
        playerPlaying: false,
        playerCurrentTime: 0,
        playerDuration: 0,
        activeSegmentIdx: -1,
        activeTrackNum: null,
        trackMuted: {},
        _trackAudios: [],
        _playerInited: false,
        _creating: null,
        _extractionES: null,
        _enrollmentES: null,
        _transcriptionES: null,

        // Step names are resolved to the active language in init() via _buildSteps()
        // (window.t is a global from shell.js, guaranteed loaded before this component).
        // Each step keeps its i18n `key` so names can be recomputed on language change.
        steps: [
            { key: 'workflow.step_upload', name: 'Upload' },
            { key: 'workflow.step_configure', name: 'Configure' },
            { key: 'workflow.step_extract', name: 'Extract' },
            { key: 'workflow.step_samples', name: 'Samples' },
            { key: 'workflow.step_enroll', name: 'Enroll' },
            { key: 'workflow.step_transcribe', name: 'Transcribe' }
        ],

        _buildSteps() {
            this.steps = this.steps.map(s => ({ ...s, name: t(s.key) }));
        },

        async init() {
            // Resolve step display names in the active language.
            this._buildSteps();

            // Check URL for existing session
            const params = new URLSearchParams(window.location.search);
            const sessionId = params.get('session');
            const step = parseInt(params.get('step')) || 1;

            if (sessionId) {
                // Restore existing session
                this.session = { id: sessionId };
                try {
                    await this.loadSession();
                    if (this.sessionNotFound) return;
                    // Restore state based on session status
                    this._syncCompletionFlags();

                    // Reconnect to running tasks
                    await this._checkRunningTasks();

                    // Restore step (with validation)
                    if (step >= 1 && step <= 6) {
                        this.currentStep = step;
                        if (step === 4) {
                            this.$nextTick(() => this.initSortables());
                        }
                    }
                } catch (error) {
                    console.error('Failed to restore session:', error);
                    this.sessionNotFound = true;
                }
            }
            // No ?session= — don't create one yet: uploadFiles() creates it lazily,
            // so merely opening the app doesn't leave an empty session behind.
            await this.loadGlobalSpeakers();
        },

        async _checkRunningTasks() {
            if (!this.session?.id) return;
            try {
                const response = await authFetch(`/api/session/${this.session.id}/tasks/status`);
                const tasks = await response.json();
                if (tasks.extract?.running) {
                    this.extracting = true;
                    this._subscribeExtraction();
                }
                if (tasks.enroll?.running) {
                    this.enrolling = true;
                    this._subscribeEnrollment();
                }
                if (tasks.transcribe?.running) {
                    this.transcribing = true;
                    this._subscribeTranscription();
                }
            } catch (error) {
                console.error('Failed to check running tasks:', error);
            }
        },

        _syncCompletionFlags() {
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
        },

        // Called when an SSE stream drops without a done/error message
        // (network blip, server restart). Resubscribe if the task is still
        // running; otherwise refresh state so the UI doesn't spin forever.
        async _recoverTask(taskType) {
            const handlers = {
                extract: { flag: 'extracting', resubscribe: () => this._subscribeExtraction() },
                enroll: { flag: 'enrolling', resubscribe: () => this._subscribeEnrollment() },
                transcribe: { flag: 'transcribing', resubscribe: () => this._subscribeTranscription() },
            }[taskType];

            await new Promise((resolve) => setTimeout(resolve, 2000));
            let running = false;
            try {
                const response = await authFetch(`/api/session/${this.session.id}/tasks/status`);
                const tasks = await response.json();
                running = !!tasks[taskType]?.running;
            } catch (error) {
                console.error('Failed to check task status:', error);
            }

            if (running) {
                handlers.resubscribe();
                return;
            }
            this[handlers.flag] = false;
            try {
                await this.loadSession();
                this._syncCompletionFlags();
            } catch (error) {
                console.error('Failed to reload session:', error);
            }
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
                const response = await authFetch('/api/session', { method: 'POST' });
                const data = await response.json();
                this.session = { id: data.session_id, tracks: [], speakers: [], samples: [] };
                this.sessionNotFound = false;
                this.currentStep = 1;
                this.updateUrl();
            } catch (error) {
                console.error('Failed to create session:', error);
            }
        },

        async loadSession() {
            if (!this.session?.id) return;
            const response = await authFetch(`/api/session/${this.session.id}`);
            if (response.status === 404) {
                this.sessionNotFound = true;
                return;
            }
            this.session = await response.json();
        },

        async loadGlobalSpeakers() {
            try {
                const response = await authFetch('/api/speakers');
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                this.globalSpeakers = await response.json();
            } catch (error) {
                console.error('Failed to load speakers:', error);
            }
        },

        isEnrolledSpeaker(name) {
            return !!name && this.globalSpeakers.some(s => s.name === name);
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

            // Step 6 always available if tracks exist
            if (step === 6) return this.session?.tracks?.length > 0;
            return false;
        },

        skipExtraction() {
            this.stopAllAudio();
            this.currentStep = 6;
            this.updateUrl();
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
            // Stop transcript player
            this.stopPlayer();
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

                if (this.currentStep === 6) {
                    if (!this.needsDiarization()) {
                        // No diarization → back to Configure
                        this.currentStep = 2;
                    } else if (!this.extractionComplete) {
                        // Diarization needed but extraction was skipped → back to Extract
                        this.currentStep = 3;
                    } else {
                        this.currentStep = 5;
                    }
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
            if (!files.length) return;

            if (!this.session?.id) {
                // Shared promise: two quick drops must not create two sessions
                this._creating ??= this.createSession();
                await this._creating;
                this._creating = null;
                if (!this.session?.id) return; // creation failed, already logged
            }

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
                            reject(new Error(xhr.responseText || t('workflow.upload_failed')));
                        }
                    });

                    xhr.addEventListener('error', () => {
                        reject(new Error(t('workflow.network_error')));
                    });

                    xhr.open('POST', `/api/session/${this.session.id}/tracks`);
                    xhr.send(formData);
                });

                await this.loadSession();
                this.uploadProgress = 100;
            } catch (error) {
                console.error('Upload failed:', error);
                alert(t('workflow.upload_failed_reason', { reason: error.message }));
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
        async updateTrackConfig(trackNum, speakerName, diarize, openSpace = false) {
            if (!this.session?.id) return;

            const params = new URLSearchParams();
            if (speakerName) params.set('speaker_name', speakerName);
            params.set('diarize', diarize);
            params.set('open_space', openSpace);

            try {
                await fetch(`/api/session/${this.session.id}/tracks/${trackNum}?${params}`, {
                    method: 'PATCH'
                });
                await this.loadSession();
            } catch (error) {
                console.error('Update failed:', error);
            }
        },

        // ---- SSE Subscribe Helpers ----

        _subscribeExtraction() {
            if (this._extractionES) this._extractionES.close();
            this.extractionLogs = [];
            this.extractionStep = 0;
            this.extractionTotal = 1;
            const eventSource = new EventSource(`/api/session/${this.session.id}/extract/stream`);
            this._extractionES = eventSource;

            eventSource.onmessage = async (event) => {
                const data = JSON.parse(event.data);
                if (data.done) {
                    eventSource.close();
                    this._extractionES = null;
                    this.extracting = false;
                    this.extractionComplete = true;
                    await this.loadSession();
                    return;
                }
                if (data.error) {
                    eventSource.close();
                    this._extractionES = null;
                    this.extracting = false;
                    this.extractionLogs.push(t('workflow.log_error', { message: data.error }));
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
                if (this._extractionES !== eventSource) return;
                eventSource.close();
                this._extractionES = null;
                this._recoverTask('extract');
            };
        },

        _subscribeEnrollment() {
            if (this._enrollmentES) this._enrollmentES.close();
            this.enrollmentLogs = [];
            const eventSource = new EventSource(`/api/session/${this.session.id}/enroll/stream`);
            this._enrollmentES = eventSource;

            eventSource.onmessage = async (event) => {
                const data = JSON.parse(event.data);
                if (data.done) {
                    eventSource.close();
                    this._enrollmentES = null;
                    this.enrolling = false;
                    this.enrollmentComplete = true;
                    this.transcriptionComplete = false;
                    await this.loadGlobalSpeakers();
                    return;
                }
                if (data.error) {
                    eventSource.close();
                    this._enrollmentES = null;
                    this.enrolling = false;
                    this.enrollmentLogs.push(t('workflow.log_error', { message: data.error }));
                    return;
                }
                if (data.message) {
                    this.enrollmentStatus = data.message;
                    this.enrollmentLogs.push(data.message);
                }
            };
            eventSource.onerror = () => {
                if (this._enrollmentES !== eventSource) return;
                eventSource.close();
                this._enrollmentES = null;
                this._recoverTask('enroll');
            };
        },

        _subscribeTranscription() {
            if (this._transcriptionES) this._transcriptionES.close();
            this.transcriptionLogs = [];
            this.transcriptionStep = 0;
            this.transcriptionTotal = 1;
            this.transcriptionProgress = 0;
            const eventSource = new EventSource(`/api/session/${this.session.id}/transcribe/stream`);
            this._transcriptionES = eventSource;
            let lastStep = 0;

            eventSource.onmessage = async (event) => {
                const data = JSON.parse(event.data);
                if (data.done) {
                    eventSource.close();
                    this._transcriptionES = null;
                    this.transcribing = false;
                    this.transcriptionComplete = true;
                    await this.loadSession();
                    return;
                }
                if (data.error) {
                    eventSource.close();
                    this._transcriptionES = null;
                    this.transcribing = false;
                    this.transcriptionLogs.push(t('workflow.log_error', { message: data.error }));
                    return;
                }
                if (data.step) {
                    if (data.step !== lastStep) {
                        this.transcriptionProgress = 0;
                        lastStep = data.step;
                    }
                    this.transcriptionStep = data.step;
                    this.transcriptionTotal = data.total;
                }
                if (data.progress !== undefined) {
                    this.transcriptionProgress = data.progress;
                }
                if (data.message) {
                    this.transcriptionStatus = data.message;
                    if (data.progress === undefined || data.progress === 0) {
                        this.transcriptionLogs.push(data.message);
                    }
                }
            };
            eventSource.onerror = () => {
                if (this._transcriptionES !== eventSource) return;
                eventSource.close();
                this._transcriptionES = null;
                this._recoverTask('transcribe');
            };
        },

        // Extraction Methods
        async startExtraction() {
            if (!this.session?.id) return;
            this.extracting = true;
            this.extractionComplete = false;
            try {
                await fetch(`/api/session/${this.session.id}/extract`, { method: 'POST' });
                this._subscribeExtraction();
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
            if (!confirm(t('workflow.confirm_delete_speaker_bin'))) return;

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
            if (!confirm(t('workflow.confirm_delete_sample'))) return;

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
            try {
                await fetch(`/api/session/${this.session.id}/enroll`, { method: 'POST' });
                this._subscribeEnrollment();
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
            try {
                await fetch(`/api/session/${this.session.id}/transcribe`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: this.whisperModel,
                        language: this.language
                    })
                });
                this._subscribeTranscription();
            } catch (error) {
                console.error('Transcription failed:', error);
                this.transcribing = false;
            }
        },

        // --- Transcript Editing Methods ---

        speakerColor(name) {
            if (!name) return 'text-gray-700';
            let hash = 0;
            for (let i = 0; i < name.length; i++) {
                hash = (hash * 31 + name.charCodeAt(i)) >>> 0;
            }
            return SPEAKER_COLORS[hash % SPEAKER_COLORS.length];
        },

        get sessionSpeakers() {
            const names = (this.session?.segments || []).map(s => s.speaker).filter(Boolean);
            return [...new Set(names)].sort();
        },

        startEditSegment(seg) {
            this.editingSegmentId = seg.id;
            this.editSegmentText = seg.text;
            this.editSegmentSpeaker = seg.speaker || '';
            this.editSegmentNewName = '';
            this.editSegmentStart = this.formatSegTime(seg.start_ms);
            this.editSegmentEnd = this.formatSegTime(seg.end_ms);
            this.insertOpen = false;
        },

        formatSegTime(ms) {
            const totalS = ms / 1000;
            const m = Math.floor(totalS / 60);
            const s = (totalS % 60).toFixed(1).padStart(4, '0');
            return `${m}:${s}`;
        },

        // "M:SS", "M:SS.s" or bare seconds; null if unparseable
        parseSegTime(str) {
            const m = /^(?:(\d+):)?(\d+(?:\.\d+)?)$/.exec(str.trim());
            if (!m) return null;
            const secs = parseFloat(m[2]);
            if (m[1] !== undefined && secs >= 60) return null;
            return Math.round((parseInt(m[1] || '0') * 60 + secs) * 1000);
        },

        cancelEditSegment() {
            this.editingSegmentId = null;
        },

        async _segmentRequest(url, options = {}) {
            this.segmentBusy = true;
            try {
                const response = await authFetch(url, options);
                if (!response.ok) {
                    const err = await response.json().catch(() => ({}));
                    const detail = typeof err.detail === 'string' ? err.detail : response.statusText;
                    alert('Edit failed: ' + detail);
                    return false;
                }
                await this.loadSession();
                // Indices shifted — let the next timeupdate recompute the active segment
                this.activeSegmentIdx = -1;
                this.activeTrackNum = null;
                return true;
            } catch (error) {
                console.error('Segment edit failed:', error);
                return false;
            } finally {
                this.segmentBusy = false;
            }
        },

        _editedSegmentPatch(seg) {
            const patch = {};
            if (this.editSegmentText !== seg.text) patch.text = this.editSegmentText;
            const speaker = this.editSegmentSpeaker === '__new__'
                ? this.editSegmentNewName.trim()
                : this.editSegmentSpeaker;
            if (speaker && speaker !== (seg.speaker || '')) patch.speaker = speaker;
            const startMs = this.parseSegTime(this.editSegmentStart);
            const endMs = this.parseSegTime(this.editSegmentEnd);
            if (startMs !== null && startMs !== seg.start_ms) patch.start_ms = startMs;
            if (endMs !== null && endMs !== seg.end_ms) patch.end_ms = endMs;
            return patch;
        },

        async saveSegmentEdit(seg) {
            if (this.parseSegTime(this.editSegmentStart) === null
                || this.parseSegTime(this.editSegmentEnd) === null) {
                alert('Invalid time — use M:SS.s');
                return;
            }
            const patch = this._editedSegmentPatch(seg);
            if (patch.text !== undefined && !patch.text.trim()) {
                alert('Segment text cannot be empty');
                return;
            }
            if (Object.keys(patch).length === 0) {
                this.editingSegmentId = null;
                return;
            }
            const ok = await this._segmentRequest(
                `/api/session/${this.session.id}/segments/${seg.id}`,
                {
                    method: 'PATCH',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(patch)
                }
            );
            if (ok) this.editingSegmentId = null;
        },

        async deleteSegment(seg) {
            if (!confirm('Delete this segment?')) return;
            await this._segmentRequest(
                `/api/session/${this.session.id}/segments/${seg.id}`,
                { method: 'DELETE' }
            );
        },

        // afterId === null inserts before the first segment
        startInsertSegment(afterId) {
            this.insertOpen = true;
            this.insertAfterId = afterId;
            this.insertText = '';
            this.insertSpeaker = '';
            this.insertNewName = '';
            this.editingSegmentId = null;
        },

        cancelInsertSegment() {
            this.insertOpen = false;
        },

        async saveInsertSegment() {
            const text = this.insertText.trim();
            if (!text) {
                alert('Segment text cannot be empty');
                return;
            }
            const speaker = this.insertSpeaker === '__new__'
                ? this.insertNewName.trim()
                : this.insertSpeaker;
            const ok = await this._segmentRequest(
                `/api/session/${this.session.id}/segments`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        after_id: this.insertAfterId,
                        text,
                        speaker: speaker || null
                    })
                }
            );
            if (ok) this.insertOpen = false;
        },

        async mergeSegmentWithNext(seg) {
            await this._segmentRequest(
                `/api/session/${this.session.id}/segments/${seg.id}/merge-next`,
                { method: 'POST' }
            );
        },

        async splitSegmentAtCursor(seg) {
            const textarea = document.getElementById('seg-edit-' + seg.id);
            if (!textarea) return;
            const offset = textarea.selectionStart;
            // Persist pending edits first so the offset refers to the stored text
            const patch = this._editedSegmentPatch(seg);
            if (patch.text !== undefined && !patch.text.trim()) {
                alert('Segment text cannot be empty');
                return;
            }
            if (Object.keys(patch).length > 0) {
                const saved = await this._segmentRequest(
                    `/api/session/${this.session.id}/segments/${seg.id}`,
                    {
                        method: 'PATCH',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(patch)
                    }
                );
                if (!saved) return;
            }
            const ok = await this._segmentRequest(
                `/api/session/${this.session.id}/segments/${seg.id}/split`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ offset })
                }
            );
            if (ok) this.editingSegmentId = null;
        },

        // --- Transcript Player Methods ---

        initPlayer() {
            if (this._playerInited) return;
            this._trackAudios = Array.from(document.querySelectorAll('audio[data-track]'));
            if (!this._trackAudios.length) return;

            const primary = this._trackAudios[0];
            primary.addEventListener('timeupdate', () => this.updatePlayerTime());
            primary.addEventListener('ended', () => {
                this.playerPlaying = false;
                this.activeSegmentIdx = -1;
                this.activeTrackNum = null;
            });
            primary.addEventListener('loadedmetadata', () => {
                this.playerDuration = primary.duration;
            });
            // If metadata already loaded
            if (primary.duration) {
                this.playerDuration = primary.duration;
            }
            this._playerInited = true;
        },

        togglePlay() {
            if (!this._trackAudios.length) this.initPlayer();
            if (!this._trackAudios.length) return;

            if (this.playerPlaying) {
                this._trackAudios.forEach(a => a.pause());
                this.playerPlaying = false;
            } else {
                this._trackAudios.forEach(a => {
                    a.currentTime = this.playerCurrentTime;
                    a.play();
                });
                this.playerPlaying = true;
            }
        },

        seekTo(timeSec) {
            this._trackAudios.forEach(a => { a.currentTime = timeSec; });
            this.playerCurrentTime = timeSec;
            this._updateActiveSegment(timeSec * 1000);
        },

        seekPlayerToPercent(event) {
            if (!this.playerDuration) return;
            const rect = event.currentTarget.getBoundingClientRect();
            const pct = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
            this.seekTo(this.playerDuration * pct);
        },

        updatePlayerTime() {
            const primary = this._trackAudios[0];
            if (!primary) return;
            this.playerCurrentTime = primary.currentTime;
            if (primary.duration) this.playerDuration = primary.duration;
            this._updateActiveSegment(primary.currentTime * 1000);
        },

        _updateActiveSegment(timeMs) {
            const segs = this.session?.segments;
            if (!segs || !segs.length) return;

            // Find segment containing timeMs
            let idx = -1;
            for (let i = 0; i < segs.length; i++) {
                if (timeMs >= segs[i].start_ms && timeMs < segs[i].end_ms) {
                    idx = i;
                    break;
                }
                // Between segments: use the one whose start is closest ahead
                if (i < segs.length - 1 && timeMs >= segs[i].end_ms && timeMs < segs[i + 1].start_ms) {
                    idx = i;
                    break;
                }
            }
            // After last segment
            if (idx === -1 && segs.length > 0 && timeMs >= segs[segs.length - 1].start_ms) {
                idx = segs.length - 1;
            }

            if (idx !== this.activeSegmentIdx) {
                this.activeSegmentIdx = idx;
                this.activeTrackNum = idx >= 0 ? segs[idx].track_num : null;
                // Auto-scroll
                if (idx >= 0) {
                    const el = document.getElementById('seg-' + idx);
                    if (el) el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
                }
            }
        },

        toggleTrackMute(trackNum) {
            this.trackMuted[trackNum] = !this.trackMuted[trackNum];
            const audio = this._trackAudios.find(a => parseInt(a.dataset.track) === trackNum);
            if (audio) audio.muted = !!this.trackMuted[trackNum];
        },

        playFromSegment(idx) {
            const segs = this.session?.segments;
            if (!segs || !segs[idx]) return;
            if (!this._trackAudios.length) this.initPlayer();

            // Click on the playing segment → pause
            if (this.activeSegmentIdx === idx && this.playerPlaying) {
                this._trackAudios.forEach(a => a.pause());
                this.playerPlaying = false;
                return;
            }

            const startSec = segs[idx].start_ms / 1000;
            this.seekTo(startSec);
            this._trackAudios.forEach(a => a.play());
            this.playerPlaying = true;
        },

        stopPlayer() {
            this._trackAudios.forEach(a => {
                a.pause();
                a.currentTime = 0;
            });
            this.playerPlaying = false;
            this.playerCurrentTime = 0;
            this.activeSegmentIdx = -1;
            this.activeTrackNum = null;
        },

        formatPlayerTime(sec) {
            if (!sec || isNaN(sec)) return '00:00';
            const m = Math.floor(sec / 60);
            const s = Math.floor(sec % 60);
            return String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0');
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
                this.copied = true;
                setTimeout(() => this.copied = false, 2000);
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

        // Called by Alpine on unmount (page switch or :key re-mount) — the only
        // place where streams from an old session get detached.
        destroy() {
            this._extractionES?.close();
            this._enrollmentES?.close();
            this._transcriptionES?.close();
            this.sortables.forEach(s => s.destroy());
            this._trackAudios.forEach(a => a.pause());
            this.$refs.samplePlayer?.pause();
        }
    }));
});
