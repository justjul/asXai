from asxai.dataset import update, process, update_payloads, arXlinks_update
from asxai.vectorDB import RerankEncoder
from datetime import datetime

from typing import Union, List
import fire
import config
from asxai.logger import get_logger
from asxai.utils import load_params

logger = get_logger(__name__, level=config.LOG_LEVEL)
params = load_params()
s2_config = params["download"]


def update_database(mode: str = 'update',
                    years: Union[int, List[int]] = datetime.now().year,
                    min_citations_per_year: float = s2_config['min_citations_per_year'],
                    fields_of_study: List[str] = s2_config['fields_of_study'],
                    update_reranker: bool = False,
                    only_new: bool = False):
    """
    Args:
        mode: "update", "push", "pull", or "nothing".
        years: A single year or a list of years to update.
        min_citations_per_year: Minimum citation threshold.
        fields_of_study: List of fields to include.
        update_reranker: Whether to retrain the reranker model.
    """

    filters = [('pdf_status', '!=', 'extracted')] if only_new else None

    if mode == "update":
        logger.info(f"Starting update for year {years}")
        update(years=years,
               min_citations_per_year=min_citations_per_year,
               fields_of_study=fields_of_study)

        # Will later include updates from arXiv but we'll then need
        # to define a way to clean up the database from old arXiv
        # papers that never got cited. The problem is that papers
        # that are not in s2 don't have citation counts for now...

        logger.info(
            f"Will now extract, embed and push new articles of {years}")
        process(years=years,
                download_extract=True,
                embed_push=True,
                filters=filters)

        logger.info(f"Updating all payloads for year {years}")
        update_payloads(years=years)

    elif mode == "push":
        logger.info(f"Pushing all articles for year {years}")

        process(years=years,
                download_extract=True,
                embed_push=True,
                filters=filters)

    elif mode == "pull":
        logger.info(f"Fetching articles for year {years}")

        update(years=years,
               min_citations_per_year=min_citations_per_year,
               fields_of_study=fields_of_study)

        logger.info(f"Updating all payloads for year {years}")
        update_payloads(years=years)

    else:
        logger.warning(f"Unsupported mode: {mode}")

    if update_reranker:
        model = RerankEncoder.load()
        if not model:
            model = RerankEncoder()
        else:
            model.lr = 1e-5  # decreased lr for fine-tuning

        model.train_reranker_from_cite(years_range=years)

    logger.info("Process completed.")


if __name__ == "__main__":
    fire.Fire(update_database)
