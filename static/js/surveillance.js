/**
 * Surveillance System Web Interface
 * Handles real-time camera streams, metrics updates, and UI interactions
 */

class SurveillanceSystem {
    constructor() {
        this.websocket = null;
        this.currentView = 'single';
        this.lastUpdate = null;
        this.startTime = Date.now();
        this.metricsTimer = null;
        this.healthTimer = null;
        this.init();
    }

    // Legacy shim: video stream setup is handled in setupEventListeners()
    setupVideoStream() {
        // No additional setup required; kept for backward compatibility
    }

    init() {
        this.setupEventListeners();
        this.updateClock();
        this.setupVideoStream();
        this.startPolling();
        // Fallback check: mark connected if the image has loaded pixels
        setInterval(() => {
            const img = document.getElementById('video-stream');
            if (img && img.naturalWidth > 0 && img.naturalHeight > 0) {
                this.updateConnectionStatus(true);
                const singleStatus = document.getElementById('single-camera-status');
                if (singleStatus) singleStatus.className = 'status-indicator status-online';
            }
        }, 2000);
        // Update clock every second
        setInterval(() => this.updateClock(), 1000);
        // Update uptime every second
        setInterval(() => this.updateUptime(), 1000);
    }

    setupEventListeners() {
        // View toggle buttons
        document.getElementById('btn-single-view').addEventListener('click', () => {
            this.switchView('single');
        });
        
        document.getElementById('btn-multi-view').addEventListener('click', () => {
            this.switchView('multi');
        });

        // Handle video stream errors
        const videoStream = document.getElementById('video-stream');
        if (videoStream) {
            videoStream.onerror = () => {
                console.log('Video stream error, attempting to reload...');
                setTimeout(() => {
                    videoStream.src = '/video/stream?' + new Date().getTime();
                }, 3000);
            };
            videoStream.onload = () => {
                // Mark camera as online and connected
                const singleStatus = document.getElementById('single-camera-status');
                if (singleStatus) singleStatus.className = 'status-indicator status-online';
                this.updateConnectionStatus(true);
            };
        }
    }

    updateClock() {
        const now = new Date();
        const timeString = now.toLocaleTimeString();
        document.getElementById('current-time').textContent = timeString;
    }

    updateUptime() {
        const uptime = Date.now() - this.startTime;
        const hours = Math.floor(uptime / (1000 * 60 * 60));
        const minutes = Math.floor((uptime % (1000 * 60 * 60)) / (1000 * 60));
        const seconds = Math.floor((uptime % (1000 * 60)) / 1000);
        
        const uptimeString = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        document.getElementById('system-uptime').textContent = uptimeString;
    }

    startPolling() {
        // Poll metrics every second
        this.metricsTimer = setInterval(async () => {
            try {
                const res = await fetch(`/api/live?t=${Date.now()}`, { cache: 'no-store' });
                if (!res.ok) throw new Error('metrics fetch failed');
                const data = await res.json();
                // Debug: log a sample every ~5s
                if (!this._lastLog || (Date.now() - this._lastLog) > 5000) {
                    console.debug('Live metrics', data);
                    this._lastLog = Date.now();
                }
                this.updateMetrics(data);
                this.updateConnectionStatus(true);
            } catch (e) {
                console.error('Metrics polling error:', e);
                this.updateConnectionStatus(false);
            }
        }, 1000);

        // Periodic health check
        this.healthTimer = setInterval(async () => {
            try {
                await fetch('/api/health', { cache: 'no-store' });
            } catch (_) { /* noop */ }
        }, 10000);

        // Periodically refresh recordings list
        this.refreshRecordings();
        setInterval(() => this.refreshRecordings(), 15000);
    }

    updateMetrics(data) {
        const peopleIn = Number(data.in) || 0;
        const peopleOut = Number(data.out) || 0;
        const occupancy = Number(data.occupancy) || 0;
        const fps = Number(data.fps) || 0;
        document.getElementById('people-in').textContent = peopleIn;
        document.getElementById('people-out').textContent = peopleOut;
        document.getElementById('occupancy').textContent = occupancy;
        document.getElementById('fps').textContent = fps.toFixed(1);
        this.lastUpdate = new Date();
        document.getElementById('last-update').textContent = this.lastUpdate.toLocaleTimeString();
        const totalDetections = (typeof data.detections === 'number') ? data.detections : (peopleIn + peopleOut);
        document.getElementById('total-detections').textContent = totalDetections;
        document.getElementById('active-cameras').textContent = '1';
    }

    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connection-status');
        if (connected) {
            statusElement.className = 'connection-status connected';
            statusElement.innerHTML = '<i class="fas fa-wifi"></i> Connected';
        } else {
            statusElement.className = 'connection-status disconnected';
            statusElement.innerHTML = '<i class="fas fa-wifi"></i> Disconnected';
        }
    }

    switchView(viewType) {
        this.currentView = viewType;
        // Update button states
        document.getElementById('btn-single-view').classList.toggle('active', viewType === 'single');
        document.getElementById('btn-multi-view').classList.toggle('active', viewType === 'multi');
        // Show/hide views
        const singleView = document.getElementById('single-camera-view');
        const multiView = document.getElementById('multi-camera-view');
        if (viewType === 'single') {
            singleView.style.display = 'block';
            multiView.style.display = 'none';
        } else {
            singleView.style.display = 'none';
            multiView.style.display = 'grid';
        }
    }

    // Removed multi-camera dynamic builders for simplified single-camera setup

    setupCameraStreams() {
        // Initialize with single view
        this.switchView('single');
        
        // Periodically check camera status
        setInterval(() => {
            this.checkCameraStatus();
        }, 5000);
    }

    async checkCameraStatus() {
        try {
            const response = await fetch('/api/health');
            if (response.ok) {
                const data = await response.json();
                // Update system status based on health check
                if (data.database_connected !== undefined) {
                    // Handle health data
                }
            }
        } catch (error) {
            console.error('Failed to check system health:', error);
        }
    }

    async refreshRecordings() {
        const container = document.getElementById('recordings-list');
        if (!container) return;
        try {
            const res = await fetch(`/api/recordings?t=${Date.now()}`, { cache: 'no-store' });
            if (!res.ok) throw new Error('failed to list recordings');
            const data = await res.json();
            const clips = Array.isArray(data.clips) ? data.clips : [];
            container.innerHTML = '';
            if (clips.length === 0) {
                container.innerHTML = '<div class="text-muted">No clips yet.</div>';
                return;
            }
            // Render first 8 clips as cards
            clips.slice(0, 8).forEach((clip) => {
                const col = document.createElement('div');
                col.className = 'col-md-3 mb-3';
                col.innerHTML = `
                    <div class="card" style="background:#1f2c3a;border:1px solid #34495e;">
                        <div class="card-body p-2">
                            <video controls preload="metadata" style="width:100%;border-radius:6px;background:#000;">
                                <source src="${clip.url}" type="video/mp4">
                                Your browser does not support the video tag.
                            </video>
                            <div class="small mt-2" style="opacity:0.8;">
                                ${clip.filename}
                            </div>
                        </div>
                    </div>`;
                container.appendChild(col);
            });
        } catch (e) {
            console.error('Failed to load recordings:', e);
            container.innerHTML = '<div class="text-danger">Failed to load recordings.</div>';
        }
    }
}

// CSS animation for flash effect
const style = document.createElement('style');
style.textContent = `
    @keyframes flash {
        0% { background-color: transparent; }
        50% { background-color: rgba(52, 152, 219, 0.3); }
        100% { background-color: transparent; }
    }
`;
document.head.appendChild(style);

// Initialize the surveillance system when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.surveillanceSystem = new SurveillanceSystem();
    const btn = document.getElementById('btn-refresh-clips');
    if (btn) btn.addEventListener('click', () => window.surveillanceSystem.refreshRecordings());
});
