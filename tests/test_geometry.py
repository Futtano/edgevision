import numpy as np
import pytest

from edgevision.geometry import preprocess


def test_known_letterbox_and_inverse():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    image[:, :, 0] = 255  # RGB red, not BGR blue.
    tensor, geometry = preprocess(image, 320)
    assert tensor.shape == (1, 3, 320, 320)
    assert tensor.dtype == np.float32 and tensor.flags.c_contiguous
    assert (geometry.left, geometry.top) == (0, 70)
    np.testing.assert_allclose(tensor[0, :, 100, 100], [1, 0, 0])
    np.testing.assert_allclose(tensor[0, :, 0, 0], [114 / 255] * 3)
    # Original box (400, 200, 800, 600), scaled by 1/4 then padded.
    assert geometry.restore((100, 120, 200, 220)) == (400, 200, 800, 600)


@pytest.mark.parametrize("shape", [(333, 1000, 3), (1000, 333, 3)])
def test_rounded_resize_uses_actual_axis_scales(shape):
    _, geometry = preprocess(np.zeros(shape, dtype=np.uint8), 320)
    h, w, _ = shape
    assert geometry.restore(
        (
            geometry.left,
            geometry.top,
            geometry.left + geometry.resized_width,
            geometry.top + geometry.resized_height,
        )
    ) == pytest.approx((0, 0, w, h))


def test_clip_and_discard_padding_boxes():
    _, geometry = preprocess(np.zeros((720, 1280, 3), dtype=np.uint8), 320)
    assert geometry.restore((-5, 60, 325, 260)) == (0, 0, 1280, 720)
    assert geometry.restore((10, 0, 20, 60)) is None


@pytest.mark.parametrize("box", [(1, 2, 0, 3), (0, 0, float("nan"), 1)])
def test_malformed_model_output_is_rejected(box):
    _, geometry = preprocess(np.zeros((64, 64, 3), dtype=np.uint8), 320)
    with pytest.raises(ValueError):
        geometry.restore(box)
