import numpy as np
import pydicom
from PIL import Image


def convert_dicom_to_png(
    dicom_file_path: str, output_dir: str, normalise: bool = True
) -> None:
    """Takes a DICOM file path and converts it to the file ending provided in the output_dir (.png or .jpg)
    see https://www.youtube.com/watch?v=k6hD0xNp2B8&list=PLQCkKRar9trMY2qJAU6H4nZQwTfZc91Oq
    """
    dicom_data = pydicom.dcmread(dicom_file_path)
    image = dicom_data.pixel_array.astype(float)
    if normalise:
        image = np.maximum(image, 0) / np.maximum(image.max(), 1)

    # Convert to 8-bit unsigned integer (0-255)
    image = (image * 255).astype(np.uint8)
    Image.fromarray(image).save(output_dir)
    return None


if __name__ == "__main__":
    # testing:
    convert_dicom_to_png(
        "data/lung_data/manifest-1725363397135/LIDC-IDRI/LIDC-IDRI-0001/01-01-2000-NA-NA-30178/3000566.000000-NA-03192/1-100.dcm",
        "/Users/newuser/Documents/ITU/master_thesis/out/test.jpg",
    )
