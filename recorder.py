import os
import cv2
import time
import threading
import shutil
import subprocess
from datetime import datetime
from typing import Optional, Tuple


class ClipRecorder:
    """
    Records short video clips while there is activity (e.g., people detected).

    Preference order for best browser playback:
    1) FFmpeg (libx264) -> MP4 (most compatible)
    2) OpenCV H.264/MP4V/XVID -> MP4/AVI (fallbacks)
    """

    def __init__(
        self,
        output_dir: str = "recordings",
        codec: str = "auto",
        idle_timeout: float = 2.0,
        ffmpeg_path: Optional[str] = None,
        ffmpeg_preset: str = "veryfast",
        ffmpeg_crf: int = 23,
    ) -> None:
        self.output_dir = output_dir
        self.codec = codec  # 'auto' selects best available
        self.idle_timeout = float(idle_timeout)
        self.ffmpeg_path = ffmpeg_path or shutil.which("ffmpeg")
        self.ffmpeg_preset = ffmpeg_preset
        self.ffmpeg_crf = int(ffmpeg_crf)

        os.makedirs(self.output_dir, exist_ok=True)

        self._writer: Optional[cv2.VideoWriter] = None
        self._is_recording: bool = False
        self._lock = threading.Lock()
        self._last_active_time: float = 0.0
        self._clip_start_time: float = 0.0
        self._current_path: Optional[str] = None
        self._current_fps: float = 0.0
        self._current_size: Optional[Tuple[int, int]] = None  # (w, h)
        self._current_ext: str = ".mp4"

        # FFmpeg subprocess related
        self._ffmpeg_proc: Optional[subprocess.Popen] = None
        self._ffmpeg_stdin = None
        self._use_ffmpeg: bool = False

    def _build_path(self, tag: Optional[str]) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{tag}_" if tag else ""
        filename = f"{base}{ts}{self._current_ext}"
        return os.path.join(self.output_dir, filename)

    def _start_ffmpeg(self, path: str, fps: float, frame_size: Tuple[int, int]) -> bool:
        if not self.ffmpeg_path:
            return False
        w, h = frame_size
        # Build ffmpeg command for raw BGR input piped via stdin
        cmd = [
            self.ffmpeg_path,
            "-hide_banner", "-loglevel", "error",
            "-y",  # overwrite
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}",
            "-r", f"{max(1.0, float(fps))}",
            "-i", "-",  # stdin
            "-an",  # no audio
            "-c:v", "libx264",
            "-preset", self.ffmpeg_preset,
            "-crf", str(self.ffmpeg_crf),
            "-movflags", "+faststart",
            path,
        ]
        try:
            self._ffmpeg_proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                bufsize=10**7,
            )
            self._ffmpeg_stdin = self._ffmpeg_proc.stdin
            self._use_ffmpeg = True
            self._current_ext = ".mp4"
            return True
        except Exception:
            self._ffmpeg_proc = None
            self._ffmpeg_stdin = None
            self._use_ffmpeg = False
            return False

    def _stop_ffmpeg(self):
        try:
            if self._ffmpeg_stdin:
                try:
                    self._ffmpeg_stdin.flush()
                except Exception:
                    pass
                try:
                    self._ffmpeg_stdin.close()
                except Exception:
                    pass
            if self._ffmpeg_proc:
                try:
                    self._ffmpeg_proc.wait(timeout=3)
                except Exception:
                    try:
                        self._ffmpeg_proc.terminate()
                    except Exception:
                        pass
        finally:
            self._ffmpeg_proc = None
            self._ffmpeg_stdin = None
            self._use_ffmpeg = False

    def _try_open(self, fourcc_str: str, fps: float, frame_size: Tuple[int, int], ext: str) -> Optional[cv2.VideoWriter]:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        fps_safe = max(1.0, float(fps))
        path = self._build_path(None).rsplit(".", 1)[0] + ext
        try:
            writer = cv2.VideoWriter(path, fourcc, fps_safe, frame_size)
            if writer is None or not writer.isOpened():
                return None
            # Save chosen extension
            self._current_ext = ext
            return writer
        except Exception:
            return None

    def _open_best_writer(self, fps: float, frame_size: Tuple[int, int], tag: Optional[str]) -> bool:
        # Prefer FFmpeg H.264 if available
        self._current_ext = ".mp4"
        path = self._build_path(tag or "camera")
        if self.codec in ("auto", "ffmpeg") and self._start_ffmpeg(path, fps, frame_size):
            self._current_path = path
            return True

        # Otherwise try OpenCV writers
        candidates = []
        if self.codec == "auto":
            candidates = [
                ("avc1", ".mp4"),
                ("H264", ".mp4"),
                ("X264", ".mp4"),
                ("mp4v", ".mp4"),
                ("XVID", ".avi"),
            ]
        elif self.codec.lower() in ("avc1", "h264", "x264"):
            candidates = [(self.codec, ".mp4")]
        elif self.codec.lower() in ("mp4v",):
            candidates = [(self.codec, ".mp4")]
        elif self.codec.lower() in ("xvid",):
            candidates = [(self.codec, ".avi")]
        else:
            candidates = [("mp4v", ".mp4"), ("XVID", ".avi")]

        for fourcc_str, ext in candidates:
            self._current_ext = ext
            w = self._try_open(fourcc_str, fps, frame_size, ext)
            if w is not None:
                # Build final path with selected extension
                path = self._build_path(tag or "camera")
                try:
                    # Close temp & reopen to final path with same fourcc
                    fourcc = w.get(cv2.VIDEOWRITER_PROP_FOURCC)
                    w.release()
                    fps_safe = max(1.0, float(fps))
                    self._writer = cv2.VideoWriter(path, int(fourcc), fps_safe, frame_size)
                    if self._writer is None or not self._writer.isOpened():
                        self._writer = None
                        continue
                    self._current_path = path
                    return True
                except Exception:
                    self._writer = None
                    continue
        return False

    def start_if_needed(self, fps: float, frame_size: Tuple[int, int], tag: Optional[str] = None) -> None:
        """Open writer if not already recording or if parameters changed."""
        with self._lock:
            need_new = False
            if not self._is_recording or (self._writer is None and not self._use_ffmpeg):
                need_new = True
            else:
                # Reopen if fps/size changed significantly
                if self._current_size != frame_size or abs(self._current_fps - float(fps)) > 5.0:
                    self._safe_close()
                    need_new = True

            if need_new:
                ok = self._open_best_writer(fps, frame_size, tag)
                if not ok:
                    # Failed to open writer, silently ignore to avoid breaking the pipeline
                    self._is_recording = False
                    self._writer = None
                    self._current_path = None
                    return

                self._is_recording = True
                self._clip_start_time = time.time()
                self._current_fps = float(fps)
                self._current_size = frame_size
            # Update last active timestamp
            self._last_active_time = time.time()

    def write(self, frame) -> None:
        with self._lock:
            if not self._is_recording:
                return
            try:
                # Ensure frame matches expected size
                h, w = frame.shape[:2]
                if self._current_size != (w, h):
                    frame = cv2.resize(frame, self._current_size)
                if self._use_ffmpeg and self._ffmpeg_stdin is not None:
                    # Write raw BGR bytes to ffmpeg stdin
                    self._ffmpeg_stdin.write(frame.tobytes())
                elif self._writer is not None:
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
            if self._use_ffmpeg:
                self._stop_ffmpeg()
            if self._writer is not None:
                self._writer.release()
        except Exception:
            pass
        finally:
            self._writer = None
            self._current_path = None
            self._current_size = None
            self._current_fps = 0.0
            self._use_ffmpeg = False
