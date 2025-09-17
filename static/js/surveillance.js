/**
 * Surveillance System Web Interface
 * Handles real-time camera streams, metrics updates, and UI interactions
 */

class SurveillanceSystem {
    constructor() {
        this.websocket = null;
        this.currentView = 'single';
        this.selectedCamera = 1;
        this.cameras = [
            { id: 1, name: 'Main Gate Camera' },
            { id: 2, name: 'Side Entrance' },
            { id: 3, name: 'Back Door' },
            { id: 4, name: 'Parking Area' },
            { id: 5, name: 'Reception' },
            { id: 6, name: 'Exit Gate' }
        ];
        this.activeCameras = new Set();
        this.lastUpdate = null;
        this.startTime = Date.now();
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.updateClock();
        this.connectWebSocket();
        this.setupCameraStreams();
        
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
        
        // Camera selection
        document.getElementById('camera-select').addEventListener('change', (e) => {
            this.selectedCamera = parseInt(e.target.value);
            this.updateSingleCameraView();
        });
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

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        try {
            this.websocket = new WebSocket(wsUrl);
            
            this.websocket.onopen = () => {
                console.log('WebSocket connected');
                this.updateConnectionStatus(true);
            };
            
            this.websocket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            };
            
            this.websocket.onclose = () => {
                console.log('WebSocket disconnected');
                this.updateConnectionStatus(false);
                
                // Attempt to reconnect after 3 seconds
                setTimeout(() => {
                    this.connectWebSocket();
                }, 3000);
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus(false);
            };
            
        } catch (error) {
            console.error('Failed to create WebSocket connection:', error);
            this.updateConnectionStatus(false);
        }
    }

    handleWebSocketMessage(data) {
        if (data.type === 'live_counts') {
            this.updateMetrics(data);
        } else if (data.type === 'camera_status') {
            this.updateCameraStatus(data);
        } else if (data.type === 'crossing_event') {
            this.handleCrossingEvent(data);
        }
    }

    updateMetrics(data) {
        document.getElementById('people-in').textContent = data.in_count || 0;
        document.getElementById('people-out').textContent = data.out_count || 0;
        document.getElementById('occupancy').textContent = data.occupancy || 0;
        document.getElementById('fps').textContent = (data.fps || 0).toFixed(1);
        
        this.lastUpdate = new Date();
        document.getElementById('last-update').textContent = this.lastUpdate.toLocaleTimeString();
        
        // Update total detections (in + out)
        const totalDetections = (data.in_count || 0) + (data.out_count || 0);
        document.getElementById('total-detections').textContent = totalDetections;
    }

    updateCameraStatus(data) {
        if (data.camera_id && data.status) {
            if (data.status === 'online') {
                this.activeCameras.add(data.camera_id);
            } else {
                this.activeCameras.delete(data.camera_id);
            }
            
            document.getElementById('active-cameras').textContent = this.activeCameras.size;
            
            // Update camera status indicators
            this.updateCameraStatusIndicator(data.camera_id, data.status === 'online');
        }
    }

    updateCameraStatusIndicator(cameraId, isOnline) {
        const statusClass = isOnline ? 'status-online' : 'status-offline';
        
        // Update single view status if it's the current camera
        if (this.currentView === 'single' && cameraId === this.selectedCamera) {
            const singleStatus = document.getElementById('single-camera-status');
            singleStatus.className = `status-indicator ${statusClass}`;
        }
        
        // Update multi view status
        const multiStatus = document.getElementById(`camera-${cameraId}-status`);
        if (multiStatus) {
            multiStatus.className = `status-indicator ${statusClass}`;
        }
    }

    handleCrossingEvent(data) {
        // Add visual feedback for crossing events
        const direction = data.direction === 'in' ? 'people-in' : 'people-out';
        const element = document.getElementById(direction);
        
        // Add flash effect
        element.style.animation = 'none';
        setTimeout(() => {
            element.style.animation = 'flash 0.5s ease-in-out';
        }, 10);
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
        
        // Show/hide camera selector
        const cameraSelector = document.getElementById('camera-selector');
        cameraSelector.style.display = viewType === 'single' ? 'block' : 'none';
        
        // Show/hide views
        const singleView = document.getElementById('single-camera-view');
        const multiView = document.getElementById('multi-camera-view');
        
        if (viewType === 'single') {
            singleView.style.display = 'block';
            multiView.style.display = 'none';
            this.updateSingleCameraView();
        } else {
            singleView.style.display = 'none';
            multiView.style.display = 'grid';
            this.updateMultiCameraView();
        }
    }

    updateSingleCameraView() {
        const camera = this.cameras.find(c => c.id === this.selectedCamera);
        if (camera) {
            document.getElementById('single-camera-name').textContent = camera.name;
            
            // Update camera stream
            const streamContainer = document.getElementById('single-camera-stream');
            const streamUrl = `/api/camera/${this.selectedCamera}/stream`;
            
            streamContainer.innerHTML = `
                <img src="${streamUrl}" 
                     alt="${camera.name}" 
                     onerror="this.style.display='none'; this.parentNode.innerHTML='<div class=\\'text-center\\'><i class=\\'fas fa-exclamation-triangle\\'></i><br>Camera Offline</div>'"
                     onload="this.style.display='block'">
            `;
        }
    }

    updateMultiCameraView() {
        const multiView = document.getElementById('multi-camera-view');
        multiView.innerHTML = '';
        
        this.cameras.forEach(camera => {
            const cameraDiv = document.createElement('div');
            cameraDiv.className = 'camera-container';
            cameraDiv.innerHTML = `
                <div class="camera-title">
                    <span><i class="fas fa-camera"></i> ${camera.name}</span>
                    <span class="status-indicator status-offline" id="camera-${camera.id}-status"></span>
                </div>
                <div class="camera-stream">
                    <img src="/api/camera/${camera.id}/stream" 
                         alt="${camera.name}"
                         onerror="this.style.display='none'; this.parentNode.innerHTML='<div class=\\'text-center\\'><i class=\\'fas fa-exclamation-triangle\\'></i><br>Camera Offline</div>'"
                         onload="this.style.display='block'">
                </div>
            `;
            multiView.appendChild(cameraDiv);
        });
    }

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
});
