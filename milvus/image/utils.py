from typing import List

import numpy as np
import cv2
from PIL import Image


def create_panoramic_view(query_image_path: str, retrieved_images: List) -> np.ndarray:
    """
    Creates a 5x5 panoramic view image from a list of images

    Params:
        query_image_path: str
            Path to the query image.
        retrieved_images: List
            List of retrieved image paths.

    Returns:
        np.ndarray: The panoramic view image.
    """
    img_height = 300
    img_width = 300
    row_count = 3

    panoramic_width = img_width * row_count
    panoramic_height = img_height * row_count
    panoramic_image = np.full(
        shape=(panoramic_height, panoramic_width, 3),
        fill_value=255,
        dtype=np.uint8
    )

    # create and resize the query image with a blue border
    query_image_null = np.full(
        shape=(panoramic_height, img_width, 3),
        fill_value=255,
        dtype=np.uint8
    )
    query_image = Image.open(query_image_path).convert("RGB")
    query_array = np.array(query_image)[:, :, ::-1]
    resized_image = cv2.resize(src=query_array, dsize=(img_width, img_height))

    border_size = 10
    blue = (255, 0, 0)  # blue color in BGR
    bordered_query_image = cv2.copyMakeBorder(
        src=resized_image,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=blue,
    )

    query_image_null[img_height * 2 : img_height * 3, 0:img_width] = cv2.resize(
        src=bordered_query_image,
        dsize=(img_width, img_height)
    )

    # add text "query" below the query image
    text = "query"
    font_scale = 1
    font_thickness = 2
    text_org = (10, img_height * 3 + 30)
    cv2.putText(
        img=query_image_null,
        text=text,
        org=text_org,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        color=blue,
        thickness=font_thickness,
        lineType=cv2.LINE_AA,
    )

    # combine the rest of the images into the panoramic view
    retrieved_imgs = [
        np.array(Image.open(img).convert("RGB"))[:, :, ::-1]
        for img in retrieved_images
    ]
    for i, image in enumerate(retrieved_imgs):
        image = cv2.resize(image, (img_width - 4, img_height - 4))
        row = i // row_count
        col = i % row_count
        start_row = row * img_height
        start_col = col * img_width

        border_size = 2
        bordered_image = cv2.copyMakeBorder(
            src=image,
            top=border_size,
            bottom=border_size,
            left=border_size,
            right=border_size,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        panoramic_image[
            start_row : start_row + img_height,
            start_col : start_col + img_width
        ] = bordered_image

        # add red index numbers to each image
        text = str(i)
        org = (start_col + 50, start_row + 30)
        (font_width, font_height), baseline = cv2.getTextSize(
            text=text,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            thickness=2
        )

        top_left = (org[0] - 48, start_row + 2)
        bottom_right = (org[0] - 48 + font_width + 5, org[1] + baseline + 5)

        cv2.rectangle(
            img=panoramic_image,
            pt1=top_left,
            pt2=bottom_right,
            color=(255, 255, 255),
            thickness=cv2.FILLED
        )
        cv2.putText(
            img=panoramic_image,
            text=text,
            org=(start_col + 10, start_row + 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    # combine the query image with the panoramic view
    panoramic_image = np.hstack([query_image_null, panoramic_image])
    return panoramic_image
