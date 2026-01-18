"""
Example script to load data from elpv-dataset and run Pixel Patrol on it.

elpv-dataset provides solar cell images as numpy arrays, so we need to:
1. Load the images from elpv-dataset
2. Save them to files organized by module type (mono/poly)
3. Use Pixel Patrol's image loader to process them
"""
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from elpv_dataset.utils import load_dataset

from pixel_patrol_base.api import (
    set_settings,
    export_project, show_report, add_paths, create_project, )
from pixel_patrol_base.core.project_settings import Settings

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    # ============================================================================
    # STEP 1: Load data from elpv-dataset
    # ============================================================================
    logger.info("Loading elpv-dataset...")
    images, defect_probs, module_types = load_dataset()
    
    logger.info(f"Loaded {len(images)} images")
    logger.info(f"Image shape: {images[0].shape}")
    logger.info(f"Module types: {set(module_types)}")

    # ============================================================================
    # STEP 2: Save images to examples/data directory organized by module type
    # ============================================================================
    # Save to examples/data so we don't have to re-download every time
    data_dir = Path(__file__).parent / "data" / "elpv_dataset"
    logger.info(f"Saving images to: {data_dir}")
    
    # Create subdirectories for each module type
    mono_dir = data_dir / "mono"
    poly_dir = data_dir / "poly"
    mono_dir.mkdir(parents=True, exist_ok=True)
    poly_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if images already exist
    existing_mono = len(list(mono_dir.glob("*.png")))
    existing_poly = len(list(poly_dir.glob("*.png")))
    
    if existing_mono + existing_poly == len(images):
        logger.info(f"Images already exist ({existing_mono} mono, {existing_poly} poly). Skipping save.")
    else:
        logger.info(f"Saving {len(images)} images...")
        # Save images
        for idx, (img, module_type, defect_prob) in enumerate(zip(images, module_types, defect_probs)):
            # Determine destination directory based on module type
            if module_type == "mono":
                dest_dir = mono_dir
            else:
                dest_dir = poly_dir
            
            # Create filename with index and defect probability
            filename = f"cell_{idx:05d}_defect_{defect_prob:.3f}.png"
            filepath = dest_dir / filename
            
            # Skip if file already exists
            if filepath.exists():
                continue
            
            # Convert numpy array to PIL Image and save
            # Images are grayscale (300x300), ensure mode is 'L'
            img_array = np.asarray(img)
            if img_array.dtype != np.uint8:
                # Normalize to 0-255 if needed
                img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8) * 255).astype(np.uint8)
            
            pil_img = Image.fromarray(img_array, mode='L')
            pil_img.save(filepath)
        
        saved_mono = len(list(mono_dir.glob("*.png")))
        saved_poly = len(list(poly_dir.glob("*.png")))
        logger.info(f"Saved {saved_mono} mono images and {saved_poly} poly images")

    # ============================================================================
    # STEP 3: Set up output directory
    # ============================================================================
    output_directory = Path(__file__).parent / "exported_projects"
    output_directory.mkdir(parents=True, exist_ok=True)
    exported_project_path = output_directory / "report_data_elpv.zip"

    # ============================================================================
    # STEP 4: Create Pixel Patrol project
    # ============================================================================
    # elpv-dataset provides images, so we use the default image loader
    # If pixel-patrol-image is installed, it will use that; otherwise falls back to basic loader
    loader_name = "bioio"  # None uses default loader, or specify "image" if available
    
    project = create_project(
        name="ELPV Solar Cells",
        base_dir=str(data_dir),
        loader=loader_name
    )

    # ============================================================================
    # STEP 5: Add paths to process (organized by module type)
    # ============================================================================
    # Add subdirectories so Pixel Patrol can group by module type
    project = add_paths(project, ["mono", "poly"])

    # ============================================================================
    # STEP 6: Configure settings
    # ============================================================================
    settings = Settings(
        cmap="viridis",
        selected_file_extensions={"png"},  # Only process PNG files (use set, not list)
        pixel_patrol_flavor="forest edition"
    )
    project = set_settings(project, settings)

    # ============================================================================
    # STEP 7: Process records and generate report
    # ============================================================================
    logger.info("Processing records...")
    project = project.process_records()

    logger.info(f"Exporting project to {exported_project_path}")
    export_project(project, exported_project_path)

    logger.info("Starting report server on port 8052...")
    logger.info(f"Images are stored in: {data_dir}")
    
    show_report(project, port=8052)  # Using different port to avoid conflicts

