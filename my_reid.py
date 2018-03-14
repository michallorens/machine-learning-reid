import os.path

import torch
from torch.autograd import Variable


class Reid(object):
    def __init__(self, model_path, feature_extractor):
        working_dir = os.path.dirname(os.path.abspath(__file__))
        self.model = torch.load(os.path.join(working_dir, model_path))
        self.feature_extractor = feature_extractor

    def __call__(self, img, n=None):
        features = Variable(self.feature_extractor(img).unsqueeze(0))
        certainty, result = self.model(features.data).sort(descending=True)
        return certainty[0][:n], result[0][:n]
