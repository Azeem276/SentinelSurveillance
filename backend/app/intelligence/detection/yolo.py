"""Ultralytics YOLO object detector with built-in ByteTrack association.

Each source owns its own detector instance so that tracker state never leaks
between cameras.
"""
from __future__ import annotations

import threading
from typing import Any

import numpy as np

from app.core.config import get_settings
from app.core.exceptions import ModelNotAvailableError
from app.core.logging import DETECTION_ERROR, get_logger
from app.intelligence.detection.base import ObjectDetector
from app.intelligence.model_registry import ensure_model, log_model_loaded
from app.intelligence.types import BBox, ObjectDetection

log = get_logger(__name__)

# Loading two Ultralytics models concurrently is not thread-safe.
_LOAD_LOCK = threading.Lock()


class YOLOObjectDetector(ObjectDetector):
    name = "yolo"
    supports_native_tracking = True

    def __init__(
        self,
        *,
        device: str | None = None,
        confidence: float | None = None,
        iou: float | None = None,
        allowed_classes: list[str] | None = None,
        tracker: str = "bytetrack.yaml",
        imgsz: int = 640,
    ) -> None:
        settings = get_settings()
        self._device = device
        self.confidence = settings.detection_confidence if confidence is None else confidence
        self.iou = settings.detection_iou if iou is None else iou
        self.allowed_classes = (
            allowed_classes if allowed_classes is not None else settings.detection_class_list
        )
        self.tracker = tracker
        self.imgsz = imgsz
        self._model: Any = None
        self._names: dict[int, str] = {}
        self._class_filter: list[int] | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------- loading
    @property
    def device(self) -> str:
        if self._device is None:
            from app.core.hardware import resolve_device

            self._device = resolve_device()
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with _LOAD_LOCK:
            if self._model is not None:
                return
            weights = ensure_model("object_detector")
            try:
                from ultralytics import YOLO  # noqa: PLC0415 (heavy optional import)
            except ImportError as exc:  # pragma: no cover
                raise ModelNotAvailableError(
                    "ultralytics is not installed. Run: pip install -r backend/requirements.txt"
                ) from exc
            try:
                model = YOLO(str(weights))
                model.to(self.device)
            except Exception as exc:
                raise ModelNotAvailableError(
                    f"failed to load object detector {weights.name}: {exc}"
                ) from exc

            self._model = model
            self._names = dict(model.names) if getattr(model, "names", None) else {}
            if self.allowed_classes:
                wanted = {c.lower() for c in self.allowed_classes}
                ids = [i for i, n in self._names.items() if str(n).lower() in wanted]
                self._class_filter = sorted(ids) or None
            log_model_loaded("object_detector", self.device, weights)

    # ----------------------------------------------------------- inference
    def _to_detections(self, results: Any) -> list[ObjectDetection]:
        out: list[ObjectDetection] = []
        if not results:
            return out
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return out

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
        clss = boxes.cls.cpu().numpy() if boxes.cls is not None else np.full(len(xyxy), -1)
        ids = boxes.id.cpu().numpy() if getattr(boxes, "id", None) is not None else None

        for i in range(len(xyxy)):
            class_id = int(clss[i])
            out.append(
                ObjectDetection(
                    bbox=BBox(*(float(v) for v in xyxy[i][:4])),
                    object_class=str(self._names.get(class_id, f"class_{class_id}")),
                    confidence=float(confs[i]),
                    class_id=class_id,
                    track_key=int(ids[i]) if ids is not None else None,
                )
            )
        return out

    def _predict_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "conf": self.confidence,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "device": self.device,
            "verbose": False,
        }
        if self._class_filter:
            kwargs["classes"] = self._class_filter
        return kwargs

    def detect(self, frame: np.ndarray) -> list[ObjectDetection]:
        if self._model is None:
            self.load()
        with self._lock:
            try:
                results = self._model.predict(source=frame, **self._predict_kwargs())
            except Exception as exc:
                log.error(DETECTION_ERROR, detector=self.name, error=str(exc))
                return []
        return self._to_detections(results)

    def track(self, frame: np.ndarray) -> list[ObjectDetection]:
        if self._model is None:
            self.load()
        with self._lock:
            try:
                results = self._model.track(
                    source=frame, persist=True, tracker=self.tracker, **self._predict_kwargs()
                )
            except Exception as exc:
                # Tracking can fail (e.g. missing tracker cfg); detection alone
                # still produces useful output, so degrade rather than die.
                log.warning(DETECTION_ERROR, detector=self.name, stage="track", error=str(exc))
                try:
                    results = self._model.predict(source=frame, **self._predict_kwargs())
                except Exception as exc2:
                    log.error(DETECTION_ERROR, detector=self.name, error=str(exc2))
                    return []
        return self._to_detections(results)

    def reset_tracker(self) -> None:
        model = self._model
        if model is None:
            return
        with self._lock:
            predictor = getattr(model, "predictor", None)
            if predictor is not None and hasattr(predictor, "trackers"):
                for tracker in predictor.trackers:
                    try:
                        tracker.reset()
                    except Exception:  # pragma: no cover - version dependent
                        pass

    @property
    def class_names(self) -> list[str]:
        return [str(v) for v in self._names.values()]
