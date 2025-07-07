from asxai.dataset import update, process, update_payloads
from asxai.dataset.download import arX_update
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
        mode: "update", "push", "pull", "arXiv" or "nothing".
        years: A single year or a list of years to update.
        min_citations_per_year: Minimum citation threshold.
        fields_of_study: List of fields to include.
        update_reranker: Whether to retrain the reranker model.
    """

    db_filter = []
    if only_new:
        db_filter = [('status', '==', 'None')]
    else:
        db_filter = [('status', '!=', 'pdf_pushed')]
    openAccess_filter = [
        ('status', '!=', 'pdf_pushed')] + [('isOpenAccess', '==', True)]

    if mode == "update":
        logger.info(f"Starting update for year {years}")
        update(years=years,
               min_citations_per_year=min_citations_per_year,
               fields_of_study=fields_of_study)

        logger.info(
            f"Will now download and extract new openAccess articles of {years}")
        process(years=years,
                download_extract=True,
                embed_push=True,
                filters=openAccess_filter)

        logger.info(
            f"Will now embed and push all other new articles of {years}")
        process(years=years,
                download_extract=False,
                embed_push=True,
                filters=db_filter)

        logger.info(f"Updating all payloads for year {years}")
        update_payloads(years=years)

    elif mode == "push":
        logger.info(
            f"Will now download and extract new openAccess articles of {years}")
        process(years=years,
                download_extract=True,
                embed_push=True,
                filters=openAccess_filter)

        logger.info(
            f"Will now embed and push all other new articles of {years}")
        process(years=years,
                download_extract=False,
                embed_push=True,
                filters=db_filter)

    elif mode == "pull":
        logger.info(f"Fetching articles for year {years}")

        update(years=years,
               min_citations_per_year=min_citations_per_year,
               fields_of_study=fields_of_study)

        logger.info(f"Updating all payloads for year {years}")
        # update_payloads(years=years)

    elif mode == "arXiv":
        logger.info(f"Fetching arXiv articles for year {years}")

        arX_update(years=years)

        # logger.info(f"Updating all payloads for year {years}")
        # update_payloads(years=years)

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
