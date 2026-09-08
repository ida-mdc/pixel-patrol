from pixel_patrol_loader_medical.plugins.loaders.nifti_loader import NiftiLoader
from pixel_patrol_loader_medical.plugins.loaders.dicom_loader import DicomLoader


def register_loader_plugins():
    return [NiftiLoader, DicomLoader]
