from pixel_patrol_labels.processors.label_processor import LabelProcessor


def register_processor_plugins():
    return [LabelProcessor]