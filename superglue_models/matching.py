import torch
from .superpoint import SuperPoint
from .superglue import SuperGlue

class Matching(torch.nn.Module):
    """Image Matching Frontend (SuperPoint + SuperGlue).

    This class orchestrates the use of the SuperPoint feature extractor 
    and the SuperGlue matching network to perform feature matching between two images.
    """

    def __init__(self, config={}):
        """Initializes the Matching module.

        Args:
            config: Dictionary containing the configuration options for SuperPoint and SuperGlue.
                    It uses 'superpoint' and 'superglue' as keys to pass respective configuration dictionaries.
        """
        super(Matching, self).__init__()
        self.superpoint = SuperPoint(config.get('superpoint', {}))
        self.superglue = SuperGlue(config.get('superglue', {}))

    def forward(self, data):
        """
        Runs the SuperPoint (if necessary) and SuperGlue.

        If keypoints are already present in the input data (with keys 'keypoints0' and 'keypoints1'),
        SuperPoint extraction is skipped. Otherwise, SuperPoint will extract keypoints, descriptors,
        and scores for both images.

        Args:
            data: Dictionary containing the input images (with keys 'image0' and 'image1').
                  Optionally, keypoints, descriptors, and scores can be provided.
        """
        pred = {}

        if 'keypoints0' not in data:
            pred0 = self.superpoint({'image': data['image0']})
            pred.update({k+'0': v for k, v in pred0.items()})

        if 'keypoints1' not in data:
            pred1 = self.superpoint({'image': data['image1']})
            pred.update({k+'1': v for k, v in pred1.items()})

        data.update(pred)

        
        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        pred.update(self.superglue(data))

        return pred
