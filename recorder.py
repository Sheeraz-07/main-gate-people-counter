import os
import cv2
import time
import threading
from datetime import datetime
from typing import Optional, Tuple


class ClipRecorder:
    """
    Records short video clips while there is activity (e.g., people detected).

    Usage pattern per frame:
    - If activity present:
        recorder.start_if_needed(fps, frame_size, tag="cam1")
        recorder.write(frame)
        recorder.mark_active()
    - Else:
        recorder.maybe_stop()
    """

    def __init__(
        self,
        output_dir: str = "recordings",
        codec: str = "mp4v",
        idle_timeout: float = 2.0,
    ) -> None:
        self.output_dir = output_dir
        self.codec = codec
        self.idle_timeout = float(idle_timeout)

        os.makedirs(self.output_dir, exist_ok=True)

        self._writer: Optional[cv2.VideoWriter] = None
        self._is_recording: bool = False
        self._lock = threading.Lock()
        self._last_active_time: float = 0.0
        self._clip_start_time: float = 0.0
        self._current_path: Optional[str] = None
        self._current_fps: float = 0.0
        self._current_size: Optional[Tuple[int, int]] = None  # (w, h)

    def _build_path(self, tag: Optional[str]) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{tag}_" if tag else ""
        filename = f"{base}{ts}.mp4"
        return os.path.join(self.output_dir, filename)

    def _open_writer(self, path: str, fps: float, frame_size: Tuple[int, int]) -> Optional[cv2.VideoWriter]:
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        # Ensure reasonable fps
        fps_safe = max(1.0, float(fps))
        try:
            writer = cv2.VideoWriter(path, fourcc, fps_safe, frame_size)
            if writer is None or not writer.isOpened():
                return None
            return writer
        except Exception:
            return None

    def start_if_needed(self, fps: float, frame_size: Tuple[int, int], tag: Optional[str] = None) -> None:
        """Open writer if not already recording or if parameters changed."""
        with self._lock:
            need_new = False
            if not self._is_recording or self._writer is None:
                need_new = True
            else:
                # Reopen if fps/size changed significantly
                if self._current_size != frame_size or abs(self._current_fps - float(fps)) > 5.0:
                    self._safe_close()
                    need_new = True

            if need_new:
                path = self._build_path(tag or "camera")
                writer = self._open_writer(path, fps, frame_size)
                if writer is None:
                    # Failed to open writer, silently ignore to avoid breaking the pipeline
                    self._is_recording = False
                    self._writer = None
                    self._current_path = None
                    return
                self._writer = writer
                self._is_recording = True
                self._clip_start_time = time.time()
                self._current_path = path
                self._current_fps = float(fps)
                self._current_size = frame_size
            # Update last active timestamp
            self._last_active_time = time.time()

    def write(self, frame) -> None:
        with self._lock:
            if self._is_recording and self._writer is not None:
                try:
                    # Ensure frame matches expected size
                    h, w = frame.shape[:2]
                    if self._current_size != (w, h):
                        # Resize to match writer size to prevent crashes
                        frame = cv2.resize(frame, self._current_size)
                    self._writer.write(frame)
                except Exception:
                    # On any write error, stop current clip to protect the pipeline
                    self._safe_close()
                    self._is_recording = False

    def mark_active(self) -> None:
        with self._lock:
            self._last_active_time = time.time()

    def maybe_stop(self) -> None:
        with self._lock:
            if not self._is_recording:
                return
            if (time.time() - self._last_active_time) >= self.idle_timeout:
                self._safe_close()
                self._is_recording = False

    def stop(self) -> None:
        with self._lock:
            if self._is_recording:
                self._safe_close()
                self._is_recording = False

    def _safe_close(self) -> None:
        try:
            if self._writer is not None:
                self._writer.release()
        except Exception:
            pass
        finally:
            self._writer = None
            self._current_path = None
            self._current_size = None
            self._current_fps = 0.0
