import cv2


def resize_frame(frame, target_size):
    """Downscale a frame to target_size (width, height) before inference.

    Running detection on a smaller frame cuts the per-frame compute cost
    substantially on constrained hardware (e.g. a Raspberry Pi), at the
    cost of some accuracy on small or distant objects.
    """
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
