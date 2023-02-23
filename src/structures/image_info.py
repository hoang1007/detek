from typing import Optional


class ImageInfo:
    def __init__(self, width: int, height: int, scale: Optional[float] = None):
        self._width = width
        self._height = height
        self._scale = scale

        if scale is not None:
            self._scaled_width, self._scaled_height = self._compute_scaled_size(scale)

    def _compute_scaled_size(self, scale: float):
        return int(self._width * scale), int(self._height * scale)

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def scaled_width(self):
        if self._scale is None:
            raise ValueError("Scale is not set")
        return self._scaled_width

    @property
    def scaled_height(self):
        if self._scale is None:
            raise ValueError("Scale is not set")
        return self._scaled_height

    @property
    def scale(self):
        if self._scale is None:
            raise ValueError("Scale is not set")
        return self._scale

    def set_scale(self, scale: float):
        self._scale = scale

        self._scaled_width, self._scaled_height = self._compute_scaled_size(scale)
