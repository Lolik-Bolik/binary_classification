import cv2
import numpy as np


def draw_text(
    img: np.ndarray,
    text: str,
    origin: tuple,
    thickness=1,
    bg_color=(128, 128, 128),
    font_scale=0.5,
):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    text_size, baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
    baseline += thickness
    text_org = np.array((origin[0], origin[1] - text_size[1]))
    cv2.rectangle(
        img,
        tuple((text_org + (0, baseline)).astype(int)),
        tuple((text_org + (text_size[0], -text_size[1])).astype(int)),
        bg_color,
        -1,
    )

    cv2.putText(
        img,
        text,
        tuple((text_org + (0, baseline / 2)).astype(int)),
        font_face,
        font_scale,
        (0, 0, 0),
        thickness,
        8,
    )

    return img


def draw_classification_legend(image: np.ndarray, class_map: dict) -> np.ndarray:
    bg_color = (255, 255, 255)
    cell_height = 40
    legend = np.zeros((image.shape[0], 300, 3), dtype=image.dtype)
    legend.fill(255)

    height = 10
    weight = 10

    draw_text(legend, "Labels:", (weight, height + 15), bg_color=bg_color)
    height += cell_height
    for class_name, class_score in class_map.items():
        draw_text(
            legend,
            f"{class_name}: {class_score}",
            (weight + 10, height),
            bg_color=bg_color,
        )
        height += cell_height
    return np.concatenate([image, legend], axis=1)