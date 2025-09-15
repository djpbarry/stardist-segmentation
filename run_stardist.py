#!/usr/bin/env python3
"""
StarDist 2D segmentation script
Takes a TIF image as input and produces a label image using StarDist 2D model
"""

import argparse
import sys
from glob import glob
from pathlib import Path

import numpy as np
from csbdeep.utils import normalize
from skimage.io import imread, imsave
from stardist.models import StarDist2D


def process_image(input_path, output_path=None):
    """
    Process a single TIF image with StarDist 2D model

    Args:
        input_path (str): Path to input TIF file
        output_path (str, optional): Path for output label image. If None, uses input name with '_labels' suffix
    """

    # Convert to Path objects for easier handling
    input_path = Path(input_path)

    # Check if input file exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Generate output path if not provided
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_labels{input_path.suffix}"
    else:
        output_path = Path(f"{output_path}/{input_path.stem}_labels.png")

    print(f"Loading image from: {input_path}")

    # Load the image
    try:
        img = imread(str(input_path))
    except Exception as e:
        raise RuntimeError(f"Failed to load image: {e}")

    print(f"Image shape: {img.shape}")
    print(f"Image dtype: {img.dtype}")

    # Normalize the image (important for StarDist)
    print("Normalizing image...")
    img_normalized = normalize(img, 1, 99.8)

    # Load the pre-trained StarDist 2D model
    print("Loading StarDist 2D model...")
    try:
        model = StarDist2D.from_pretrained('2D_versatile_fluo')
    except Exception as e:
        print(f"Failed to load default model, trying alternative: {e}")
        try:
            model = StarDist2D.from_pretrained('2D_versatile_he')
        except Exception as e2:
            raise RuntimeError(f"Failed to load any pre-trained model: {e2}")

    print("Running StarDist prediction...")

    # Run prediction
    try:
        labels, details = model.predict_instances(img_normalized)
    except Exception as e:
        raise RuntimeError(f"StarDist prediction failed: {e}")

    print(f"Detected {len(details['points'])} objects")
    print(f"Label image shape: {labels.shape}")
    print(f"Label image dtype: {labels.dtype}")
    print(f"Number of unique labels: {len(np.unique(labels))}")

    # Save the label image
    print(f"Saving label image to: {output_path}")
    try:
        # Ensure labels are in appropriate format for saving
        if labels.dtype != np.uint16 and np.max(labels) < 65535:
            labels = labels.astype(np.uint16)
        elif np.max(labels) >= 65535:
            print("Warning: Many objects detected, using 32-bit labels")
            labels = labels.astype(np.uint32)

        imsave(str(output_path), labels, check_contrast=False)
    except Exception as e:
        raise RuntimeError(f"Failed to save label image: {e}")

    print("Processing completed successfully!")

    return labels, details


def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run StarDist 2D segmentation on a TIF image',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stardist_segment.py input.tif
  python stardist_segment.py input.tif -o output_labels.tif
  python stardist_segment.py /path/to/image.tif -o /path/to/output/
        """
    )

    parser.add_argument('-i', '--input', help='Path to input TIF image directory', type=str)
    parser.add_argument('-f', '--from_index', help='Start file index', type=int)
    parser.add_argument('-t', '--to_index', help='Last file index', type=int)
    parser.add_argument('-o', '--output', help='Path for output label image (optional)', type=str)
    parser.add_argument('--info', action='store_true', help='Print additional information about detected objects')
    args = parser.parse_args()

    file_list = glob(f'{args.input}/*.tif')

    try:
        for i in range(args.from_index, args.to_index):
            labels, details = process_image(file_list[i], args.output)
            if args.info:
                print("\n=== Additional Information ===")
                print(f"Object centers: {len(details['points'])} points")
                print(f"Probability scores range: {np.min(details['prob']):.3f} - {np.max(details['prob']):.3f}")
                print(f"Label range: {np.min(labels)} - {np.max(labels)}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
