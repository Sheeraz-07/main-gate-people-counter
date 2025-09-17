"""
Flask REST API and MJPEG video streaming for people counting system.
Serves the web frontend, live counts, health, and video stream.
"""
import asyncio
import logging
import time
from datetime import datetime
from typing import Callable, Dict, Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, make_response, send_from_directory, abort
from flask_cors import CORS

from db import CounterDB

logger = logging.getLogger(__name__)


class PeopleCounterAPI:
    """Flask-based API server for the people counting system."""

    def __init__(self, db_path: str = "counts.db"):
        # Flask app with static and template folders
        self.app = Flask(
            __name__, static_folder="static", template_folder="templates"
        )
        CORS(self.app)

        # Database (async aiosqlite wrapper)
        self.db = CounterDB(db_path)

        # State
        self.start_time = datetime.now()
        self.current_counts: Dict[str, int] = {"in": 0, "out": 0, "occupancy": 0}
        self.current_fps: float = 0.0
        self.current_detections: int = 0
        self.current_tracks: int = 0
        self.system_status = "starting"

        # Frame source callable provided by main system
        self.frame_source: Optional[Callable[[], Optional[np.ndarray]]] = None

        # DB readiness flag
        self._db_ready = False

        # Register routes
        self._register_routes()
        logger.info("People Counter Flask API initialized")

    def set_frame_source(self, frame_source: Callable[[], Optional[np.ndarray]]):
        """Set a callable that returns the latest processed frame (numpy array)."""
        self.frame_source = frame_source

    def set_status(self, status: str):
        self.system_status = status
        logger.info(f"System status updated to: {status}")

    def _register_routes(self):
        app = self.app

        # Note: Flask 3 removed before_first_request; DB will be initialized in run()

        @app.route("/")
        def index():
            return render_template("index.html")

        @app.route("/api/live")
        def api_live():
            # Always return current live values from memory for instant UI updates
            live_counts = self.current_counts or {"in": 0, "out": 0, "occupancy": 0}
            payload = {
                "in": int(live_counts.get("in", 0)),
                "out": int(live_counts.get("out", 0)),
                "occupancy": int(live_counts.get("occupancy", 0)),
                "fps": float(self.current_fps),
                "detections": int(self.current_detections),
                "tracks": int(self.current_tracks),
                "timestamp": datetime.now().isoformat(),
                "status": self.system_status,
            }
            # Optionally include today's totals from DB
            if self._db_ready:
                try:
                    totals = asyncio.run(self.db.get_today_counts())
                    payload["today"] = {
                        "in": int(totals.get("in", 0)),
                        "out": int(totals.get("out", 0)),
                        "occupancy": int(totals.get("occupancy", 0)),
                    }
                except Exception:
                    pass
            resp = make_response(jsonify(payload))
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
            return resp

        @app.route("/api/health")
        def api_health():
            try:
                if self._db_ready:
                    asyncio.run(self.db.get_today_counts())
                    db_connected = True
                else:
                    db_connected = False
            except Exception:
                db_connected = False
            uptime = (datetime.now() - self.start_time).total_seconds()
            payload = {
                "status": self.system_status,
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": uptime,
                "database_connected": db_connected,
            }
            resp = make_response(jsonify(payload))
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
            return resp

        @app.route("/video/stream")
        def video_stream():
            return Response(
                self._generate_frames(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @app.route("/api/recordings")
        def list_recordings():
            """Return JSON list of available video clips in the recordings directory."""
            try:
                import os
                from pathlib import Path
                rec_dir = Path("recordings")
                rec_dir.mkdir(exist_ok=True)
                entries = []
                for name in os.listdir(rec_dir):
                    path = rec_dir / name
                    if not path.is_file():
                        continue
                    # Only allow common video extensions
                    if path.suffix.lower() not in [".mp4", ".avi", ".mov", ".mkv"]:
                        continue
                    stat = path.stat()
                    entries.append({
                        "filename": name,
                        "size": stat.st_size,
                        "mtime": stat.st_mtime,
                        "url": f"/recordings/{name}",
                    })
                # Sort newest first
                entries.sort(key=lambda e: e["mtime"], reverse=True)
                resp = make_response(jsonify({"clips": entries}))
                resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
                resp.headers["Pragma"] = "no-cache"
                resp.headers["Expires"] = "0"
                return resp
            except Exception as e:
                logger.error(f"Failed to list recordings: {e}")
                return jsonify({"clips": []}), 200

        @app.route("/recordings/<path:filename>")
        def get_recording(filename: str):
            """Serve a recording file safely from the recordings directory."""
            from pathlib import Path
            rec_dir = Path("recordings").resolve()
            try:
                # Basic path traversal protection
                safe_path = (rec_dir / filename).resolve()
                if not str(safe_path).startswith(str(rec_dir)):
                    abort(404)
                if not safe_path.exists() or not safe_path.is_file():
                    abort(404)
                return send_from_directory(rec_dir, safe_path.name, as_attachment=False)
            except Exception as e:
                logger.error(f"Failed to serve recording '{filename}': {e}")
                abort(404)

    def _generate_frames(self):
        """Generator that yields JPEG-encoded frames as MJPEG stream."""
        while True:
            frame = None
            try:
                if self.frame_source is not None:
                    frame = self.frame_source()
            except Exception as e:
                logger.error(f"Frame source error: {e}")

            if frame is None:
                # Placeholder if no frame yet
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    placeholder,
                    "Initializing...",
                    (180, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                ret, buffer = cv2.imencode(".jpg", placeholder, [cv2.IMWRITE_JPEG_QUALITY, 75])
            else:
                # Reduce JPEG quality a bit to lower CPU and bandwidth
                ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])

            if ret:
                jpg = buffer.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                )

            # Cap MJPEG at ~12 FPS to reduce CPU load
            time.sleep(1 / 12.0)

    async def update_counts(self, counts: Dict[str, int], fps: float = 0.0, detections: int = 0, tracks: int = 0):
        """Update current counts, FPS, and instantaneous detections/tracks."""
        self.current_counts = counts or {"in": 0, "out": 0, "occupancy": 0}
        self.current_fps = float(fps)
        self.current_detections = int(detections)
        self.current_tracks = int(tracks)

    async def broadcast_crossing(self, crossing_event: Dict):
        # No-op for Flask (no WebSocket). Reserved for future Server-Sent Events.
        return

    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        logger.info(f"Starting Flask API server on {host}:{port}")
        # Initialize database before serving requests
        try:
            asyncio.run(self.db.init_db())
            self._db_ready = True
            self.system_status = "active"
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            self._db_ready = False
            self.system_status = "degraded"
        # Disable reloader in thread, enable threaded handling
        self.app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)
