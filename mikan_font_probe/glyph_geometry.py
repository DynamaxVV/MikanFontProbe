"""Width-reduced glyph geometry and conservative raster-scale detection."""
import cv2
import numpy as np


def thin(ink):
    mask = (ink >= 128).astype(np.uint8)
    for _ in range(64):
        changed = False
        for phase in (0, 1):
            p = np.pad(mask, 1)
            n = [p[:-2,1:-1], p[:-2,2:], p[1:-1,2:], p[2:,2:],
                 p[2:,1:-1], p[2:,:-2], p[1:-1,:-2], p[:-2,:-2]]
            count = sum(n)
            transitions = sum((n[i] == 0) & (n[(i+1)%8] == 1) for i in range(8))
            if phase == 0:
                constraint = (n[0]*n[2]*n[4] == 0) & (n[2]*n[4]*n[6] == 0)
            else:
                constraint = (n[0]*n[2]*n[6] == 0) & (n[0]*n[4]*n[6] == 0)
            remove = (mask == 1) & (count >= 2) & (count <= 6) & (transitions == 1) & constraint
            changed |= bool(remove.any())
            mask[remove] = 0
        if not changed:
            break
    return mask


def shape_loss(a, b):
    if not a.any() or not b.any():
        return 1.
    da = cv2.distanceTransform(1-a, cv2.DIST_L2, 3)
    db = cv2.distanceTransform(1-b, cv2.DIST_L2, 3)
    return min(1., (float(da[b > 0].mean())+float(db[a > 0].mean()))/8)


def repeated_scale(gray):
    """Require periodic exact row AND column duplication; never infer from size."""
    factors = []
    for axis in (0, 1):
        rows = np.moveaxis(gray, axis, 0).reshape(gray.shape[axis], -1)
        duplicate = np.all(rows[1:] == rows[:-1], axis=1)
        found = 1
        if len(duplicate) >= 15:
            for factor in (4, 3, 2):
                for offset in range(factor):
                    inside = (np.arange(len(duplicate))+offset+1) % factor != 0
                    if duplicate[inside].mean() >= .98 and duplicate[~inside].mean() <= .35:
                        found = factor
                        break
                if found > 1:
                    break
        factors.append(found)
    return factors[0] if factors[0] == factors[1] else 1


def effective_resolution(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Blank glyph")
    native = int(max(xs.max()-xs.min()+1, ys.max()-ys.min()+1))
    scale = repeated_scale(gray)
    return {"native": native, "repeat_scale": scale, "effective": max(1, round(native/scale))}
