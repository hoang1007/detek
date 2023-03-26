from typing import List, Optional
from torch import Tensor


class DetResult:
    def __init__(self, bboxes: Tensor, scores: Tensor, labels: Tensor):
        self._image = None
        self._classes = None
        self._bboxes = bboxes
        self._scores = scores
        self._labels = labels

    def set_image(self, image: Tensor):
        self._image = image
    
    def set_classes(self, classes: List[str]):
        self._classes = classes

    @property
    def bboxes(self) -> Tensor:
        return self._bboxes.clone()

    @property
    def scores(self) -> Tensor:
        return self._scores.clone()

    @property
    def labels(self) -> Tensor:
        return self._labels.clone()

    @property
    def image(self) -> Tensor:
        if self._image is None:
            raise ValueError("Image is not set")
        return self._image.clone()
    
    def visualize(self, backend: str = "cv2"):
        """
        Visualize the detection result

        Args:
            backend (str, optional): Backend to use `cv2`, `matplot` or `none`. Defaults to "cv2".
        """
        if self._image is None:
            raise ValueError("Image is not set")
        from torchvision.utils import draw_bounding_boxes
        labels = None
        if self._classes is not None:
            labels = [self._classes[label] for label in self._labels.tolist()]
        image = draw_bounding_boxes(self._image, self.bboxes, labels=labels)

        if backend == "none":
            return image
        elif backend == "cv2":
            import cv2
            cv2.imshow("image", image[[2, 1, 0]].moveaxis(0, -1).numpy())
            cv2.waitKey(0)
        elif backend == "matplot":
            import matplotlib.pyplot as plt
            plt.imshow(image.moveaxis(0, -1).numpy())
            plt.show()
