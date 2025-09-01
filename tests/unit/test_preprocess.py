import pytest
from PIL import Image
import numpy as np
from ml.inference.preproc import preprocess

def test_preprocess_output_shape_and_type():
    """
    Tests that the preprocess function returns a numpy array
    with the correct shape and data type.
    """
    # Create a dummy 3-channel image
    dummy_image = Image.new('RGB', (300, 400), color = 'red')

    processed_image = preprocess(dummy_image, size=224)

    # Check shape: [batch, channels, height, width]
    assert processed_image.shape == (1, 3, 224, 224)

    # Check data type
    assert processed_image.dtype == np.float32

def test_preprocess_normalization():
    """
    Tests that the normalization is applied correctly.
    A pure white image should have values based on the inverted mean.
    """
    # Create a pure white image (pixel values are 255)
    white_image = Image.new('RGB', (224, 224), color = 'white')

    processed_image = preprocess(white_image)

    # After dividing by 255, all values are 1.0
    # The formula is (1.0 - mean) / std
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    expected_values = (1.0 - mean) / std

    # Check that the means of the processed channels match the expected values
    # We check the mean because the exact values might have tiny floating point differences
    assert np.isclose(processed_image[0, 0, :, :].mean(), expected_values[0])
    assert np.isclose(processed_image[0, 1, :, :].mean(), expected_values[1])
    assert np.isclose(processed_image[0, 2, :, :].mean(), expected_values[2])
