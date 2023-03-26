from torch import Tensor

class DetResult:
    def __init__(self, bboxes: Tensor, scores: Tensor, labels: Tensor):
        self._bboxes = bboxes
        self._scores = scores
        self._labels = labels
    
    @property
    def bboxes(self) -> Tensor:
        return self._bboxes.clone()

    @property
    def scores(self) -> Tensor:
        return self._scores.clone()
    
    @property
    def labels(self) -> Tensor:
        return self._labels.clone()
