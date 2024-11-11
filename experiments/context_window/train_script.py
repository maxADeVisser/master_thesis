from model.train import train_model
from project_config import pipeline_config

CONTEXT_WINDOW_SIZES = pipeline_config.dataset.image_dims

from utils.logger_setup import logger


def train_context_models(context_window_sizes: list[int]) -> None:
    logger.info(f"\nTraining models with context window sizes:\n{context_window_sizes}")
    for context_size in context_window_sizes:
        train_model(
            model_name=f"contextwindow{context_size}",
            context_window_size=context_size,
            cross_validation=True,
        )


if __name__ == "__main__":
    train_context_models(CONTEXT_WINDOW_SIZES)
