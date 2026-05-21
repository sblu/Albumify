import random

from PIL import Image, ImageChops

from albumify.transforms import PairedTransformConfig, paired_transform


def make_cover(size=(400, 300), color=(255, 0, 0)) -> Image.Image:
    return Image.new("RGB", size, color)


def make_label(size=(400, 300)) -> Image.Image:
    img = Image.new("L", size, 255)
    # Draw a black pixel grid that should survive the transform deterministically.
    px = img.load()
    for x in range(0, size[0], 20):
        for y in range(0, size[1]):
            px[x, y] = 0
    return img


def test_paired_transform_output_sizes_match():
    cfg = PairedTransformConfig(out_size=128, resize_short_to=160)
    rng = random.Random(0)
    c, l = paired_transform(make_cover(), make_label(), cfg=cfg, rng=rng, train=True)
    assert c.size == (128, 128)
    assert l.size == (128, 128)
    assert c.mode == "RGB"
    assert l.mode == "L"


def test_paired_transform_geometric_alignment_after_flip():
    """Cover + label go through the same flip; geometry stays paired."""
    cfg = PairedTransformConfig(
        out_size=64, resize_short_to=64,
        hflip_prob=1.0, enable_jitter=False,
    )
    cover = Image.new("RGB", (64, 64))
    label = Image.new("L", (64, 64))
    # Paint left half of both red/black, right half white.
    for x in range(64):
        for y in range(64):
            if x < 32:
                cover.putpixel((x, y), (255, 0, 0))
                label.putpixel((x, y), 0)
            else:
                cover.putpixel((x, y), (255, 255, 255))
                label.putpixel((x, y), 255)
    c, l = paired_transform(cover, label, cfg=cfg, rng=random.Random(0), train=True)
    # After hflip, both should have left half white and right half painted.
    assert c.getpixel((5, 5)) == (255, 255, 255)
    assert c.getpixel((59, 5)) == (255, 0, 0)
    assert l.getpixel((5, 5)) == 255
    assert l.getpixel((59, 5)) == 0


def test_paired_transform_eval_is_deterministic():
    cfg = PairedTransformConfig(out_size=128, resize_short_to=160)
    c1, l1 = paired_transform(make_cover(), make_label(), cfg=cfg, train=False)
    c2, l2 = paired_transform(make_cover(), make_label(), cfg=cfg, train=False)
    assert ImageChops.difference(c1, c2).getbbox() is None
    assert ImageChops.difference(l1, l2).getbbox() is None


def test_paired_transform_train_same_seed_same_output():
    cfg = PairedTransformConfig(out_size=64, resize_short_to=80, enable_jitter=True)
    c1, l1 = paired_transform(make_cover(), make_label(),
                              cfg=cfg, rng=random.Random(123), train=True)
    c2, l2 = paired_transform(make_cover(), make_label(),
                              cfg=cfg, rng=random.Random(123), train=True)
    assert ImageChops.difference(c1, c2).getbbox() is None
    assert ImageChops.difference(l1, l2).getbbox() is None


def test_paired_transform_jitter_does_not_touch_label():
    """Photometric jitter must NEVER alter the label."""
    cfg = PairedTransformConfig(
        out_size=64, resize_short_to=64,
        hflip_prob=0.0, enable_jitter=True,
        jitter_brightness=0.5, jitter_contrast=0.5, jitter_saturation=0.5,
    )
    label = make_label((64, 64))
    _, l_t = paired_transform(make_cover((64, 64)), label,
                              cfg=cfg, rng=random.Random(99), train=True)
    # Same identity transform was applied (mode + size preserved). The label
    # values should be identical to the original (post mode collapse).
    assert ImageChops.difference(label.convert("L"), l_t).getbbox() is None


def test_paired_transform_rectangular_input_centered_in_eval():
    """Rectangular input should be resized + center-cropped at eval time."""
    cfg = PairedTransformConfig(out_size=64, resize_short_to=80)
    cover = make_cover((400, 100))  # wide rectangle
    label = make_label((400, 100))
    c, l = paired_transform(cover, label, cfg=cfg, train=False)
    assert c.size == (64, 64)
    assert l.size == (64, 64)
