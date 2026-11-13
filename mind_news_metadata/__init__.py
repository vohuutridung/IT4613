"""Independent MIND-large title/abstract and metadata field encoder.

Keep package import lightweight: stage-01 preparation can run in the project's
Python environment without importing TensorFlow, which is isolated separately.
Import the model from ``mind_news_metadata.field_encoder`` when needed.
"""
