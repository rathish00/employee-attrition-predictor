"""Custom exceptions so callers can catch specific, meaningful failures
instead of bare Exception / KeyError / ValueError from deep in the stack."""


class AttritionPredictorError(Exception):
    """Base class for all errors raised by this package."""


class DataValidationError(AttritionPredictorError):
    """Raised when input data doesn't match the expected schema."""


class ModelNotFoundError(AttritionPredictorError):
    """Raised when a trained model artifact is requested but doesn't exist on disk."""


class ModelNotTrainedError(AttritionPredictorError):
    """Raised when inference is attempted before training/loading a model."""
