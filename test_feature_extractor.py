import unittest
import torch
import numpy as np
from torchvision import transforms
from model import ConvAutoencoder_v2
from feature_extractor import get_latent_features, get_latent_features1D
from PIL import Image
import os

class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        # Setup model and dummy image
        self.model = ConvAutoencoder_v2().to('cpu')
        self.model.eval()

        # Dummy image path
        self.test_image = 'dataset/sample.jpg'
        if not os.path.exists('dataset'):
            os.makedirs('dataset')
        Image.new('RGB', (256, 256), color='white').save(self.test_image)

        # Transformation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def test_latent_features_shape(self):
        features = get_latent_features(self.model, [self.test_image], self.transform)
        self.assertEqual(features.shape, (1, 256, 8, 8))

    def test_latent_features1d_shape(self):
        features1d = get_latent_features1D(self.model, [self.test_image], self.transform)
        self.assertEqual(features1d.shape[0], 1)
        self.assertTrue(features1d.shape[1] > 0)

    def tearDown(self):
        os.remove(self.test_image)

if __name__ == '__main__':
    unittest.main()

