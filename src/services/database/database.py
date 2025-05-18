from dataset import update, process, update_payloads, arXlinks_update
from vectorDB import RerankEncoder
from datetime import datetime

from typing import Union, List
import argparse
import logging
from src.logger import get_logger
from src.utils import load_params

logger = get_logger(__name__, level=logging.INFO)
params = load_params()
s2_config = params["download"]


def update_database(mode: str = 'update',
                    years: Union[int, List[int]] = datetime.now().year,
                    min_citations_per_year: float = s2_config['min_citations_per_year'],
                    fields_of_study: List[str] = s2_config['fields_of_study'],
                    update_reranker: bool = False):
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
        process(download_extract=True,
                embed_push=True,)
        # filters=[('pdf_status', '!=', 'extracted')])

        logger.info(f"Updating all payloads for year {years}")
        update_payloads(years=years)

    elif mode == "push":
        logger.info(f"Pushing all articles for year {years}")

        process(download_extract=True,
                embed_push=True)

    elif mode == "pull":
        logger.info(f"Fetching articles for year {years}")

        update(years=years,
               min_citations_per_year=min_citations_per_year,
               fields_of_study=fields_of_study)

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


def main():
    parser = argparse.ArgumentParser(description="Update database")
    parser.add_argument("mode", choices=["update", "push", "pull", "nothing"],
                        help="Type of update to perform")
    parser.add_argument("--years", nargs="*", type=int,
                        default=datetime.now().year,
                        help="Optional list of years to process")
    parser.add_argument("--fields", nargs="*", type=str,
                        default=["Computer Science", "Biology"],
                        help="List of fields of research")
    parser.add_argument("--citation", type=float,
                        default=1,
                        help="Min rate of citations per year")
    parser.add_argument("--update-reranker", action="store_true",
                        help="If set, retrain and update the reranker model")
    args = parser.parse_args()

    update_database(mode=args.mode,
                    years=args.years,
                    fields_of_study=args.fields,
                    min_citations_per_year=args.citation,
                    update_reranker=args.update_reranker)


if __name__ == "__main__":
    main()
