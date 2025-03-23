import re
import multiprocessing
from statistics import mode
from typing import List, Optional, Tuple, Set

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTTextLine, LTChar, LTTextBox
from pdfminer.pdfpage import PDFPage
import logging
from src.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)
pdfminer_logger = logging.getLogger('pdfminer')
pdfminer_logger.setLevel(logging.ERROR)


def get_pdf_num_pages(pdf_path: str) -> int:
    """Return total number of pages in a PDF."""

    with open(pdf_path, "rb") as f:
        return sum(1 for _ in PDFPage.get_pages(f))


def is_valid_page(pdf_path: str, page_idx: int) -> bool:
    """Check if a specific page in the PDF can be parsed without error."""

    try:
        for _ in extract_pages(pdf_path, page_numbers=[page_idx]):
            return True
    except Exception as e:
        return False


def get_valid_pages(
        pdf_path: str,
        timeout: float = 1.0,
        num_processes: int = 1) -> List[int]:
    """Return a list of valid page indices."""

    valid_pages = []
    try:
        n_pages = get_pdf_num_pages(pdf_path)
        with multiprocessing.Pool(processes=num_processes) as pool:
            async_results = [pool.apply_async(is_valid_page, (pdf_path, i))
                             for i in range(n_pages)]
            for i, result in enumerate(async_results):
                try:
                    if result.get(timeout=timeout):
                        valid_pages.append(i)
                except multiprocessing.TimeoutError:
                    logger.warning(
                        f"Timeout on page {i} out of {n_pages} for {pdf_path}")
                    pool.terminate()
                    break
    except Exception as e:
        logger.warning(f"Error reading PDF {pdf_path}: {e}")

    return valid_pages


def pdf_main_fontsize(
        pdf_path: str,
        valid_pages: Optional[List[int]] = None) -> Optional[float]:
    """Estimate the main font size used in the document."""

    try:
        sizes = [
            character.size
            for page_layout in extract_pages(pdf_path, page_numbers=valid_pages)
            for element in page_layout
            if isinstance(element, LTTextContainer)
            for text_line in element
            if isinstance(text_line, LTTextLine)
            for character in text_line
            if isinstance(character, LTChar)]
        return mode(sizes) if sizes else None
    except Exception as e:
        logger.warning(f"Font size extraction failed for {pdf_path}: {e}")
        return None


def _filter_text_block(element) -> Optional[Tuple[str, List[float]]]:
    filtered_text, sizes = [], []
    for text_line in element:
        if isinstance(text_line, LTTextLine):
            sizes.extend([
                character.size
                for character in text_line
                if isinstance(character, LTChar)])
            line_text = [character.get_text() for character in text_line]
            line_text = "".join(line_text).strip()
            if not line_text.isdigit():
                filtered_text.append(line_text)
            filtered_text.append("\n")

    cleaned_text = re.sub(r"\(cid:[^\)]+\)", "",
                          "".join(filtered_text).strip())
    return cleaned_text, sizes


def _get_markers():
    last_section_markers = {
        "acknowledgment",
        "acknowledgement",
        "acknowlegment",
        "reference",
    }
    month_markers = {
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "november",
        "december",
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "nov",
        "dec",
    }
    year_markers = {str(y) for y in range(1900, 2030)}
    section_list = {
        "introduction",
        "results",
        "discussion",
        "conclusions",
        "methods",
        "materials",
        "experimental",
        "materials and methods",
        "experimental procedure",
        "related work",
        "i.",
        "ii.",
        "iii.",
        "iv.",
        "v.",
        "vi.",
    }
    return last_section_markers, month_markers, year_markers, section_list


def _extract_sections_pdfminer(
        pdf_path: str,
        authorlist: str = None,
        valid_pages: Optional[List[int]] = None,
        possible_section_headings: Optional[Set] = None,
        size_threshold: Optional[float] = None) -> dict:
    author_last_names = (
        [name.split()[-1] for name in authorlist.split(",") if name.split()]
        if authorlist
        else None)

    last_section_names, month_markers, year_markers, section_list = _get_markers()

    section_list = possible_section_headings or section_list

    size_threshold = size_threshold if size_threshold is not None else 0.9

    sections = {
        "pdf_abstract": None,
        "pdf_introduction": None,
        "pdf_results": None,
        "pdf_discussion": None,
        "pdf_methods": None,
        "pdf_text": None,
        "author_list": authorlist}

    if valid_pages == 0:
        return sections

    # Determine the most common font size (mode)
    size_mode = pdf_main_fontsize(pdf_path, valid_pages)

    page_width, page_height = 595, 842
    nword_abstract_th, nword_sections_th = 30, 30
    text_blocks, tag_blocks, nwords_in_blocks = [], [], []
    section_to_find, section_heading, tag = "AUTHORS", "UNDEFINED", "UNDEFINED"
    reached_end = False
    if size_mode is not None:
        for p, page_layout in enumerate(extract_pages(pdf_path, page_numbers=valid_pages)):
            if reached_end:
                break
            for element in page_layout:
                if not isinstance(element, LTTextBox):
                    continue

                x0, y0, x1, y1 = element.bbox
                if not (y0 > 0.05 * page_height
                        and y1 < 0.95 * page_height
                        and x0 > 0.05 * page_width
                        and x1 < 0.95 * page_width):
                    continue

                filtered_text, sizes = _filter_text_block(element)

                word_list = re.split(r"[\n\s]+", filtered_text.lower().strip())
                nwords = len(word_list)

                if any(end_section in " ".join(word_list[:3])
                       for end_section in last_section_names):
                    reached_end = True
                    continue

                # removing everything before the author block as well as the correspondance fields
                if p <= 1 and author_last_names:
                    nauthors_detected = sum(
                        lastname.lower() in " ".join(word_list)
                        for lastname in author_last_names)
                    if (nauthors_detected >= 0.5 * len(author_last_names)
                        and y0 > 0.3 * page_height
                            and section_to_find == "AUTHORS"):
                        text_blocks, tag_blocks, nwords_in_blocks = [], [], []
                        section_to_find = "ABSTRACT"
                        filtered_text = []
                    if nauthors_detected > 0 and ("@" in filtered_text or "correspond" in filtered_text):
                        filtered_text = []

                # removing blocks likely headers with publication date
                if (any([m in word_list for m in month_markers])
                    and any([y in word_list for y in year_markers])
                        and nwords < 10):
                    continue

                # removing figure captions
                if any([figname in word_list[0]
                        for figname in ["fig", "figure", "table", "image"]]):
                    continue

                # removing previous block if likely a header but not followed by capitalized paragraph
                if (filtered_text
                    and not filtered_text[0].isupper()
                    and nwords_in_blocks
                        and nwords_in_blocks[-1] <= 3):
                    text_blocks.pop()
                    tag_blocks.pop()
                    nwords_in_blocks.pop()
                elif (filtered_text
                      and filtered_text.strip()
                      and filtered_text.strip()[0].isupper()
                      and nwords_in_blocks
                      and nwords_in_blocks[-1] <= 3
                      and text_blocks
                      and any(h in re.sub(r"[\n\s]+", " ", text_blocks[-1].lower())
                              for h in section_list)):
                    section_heading = "".join([w.upper()
                                               for w in re.sub(r"[\d.]", "", text_blocks[-1])])
                    if nwords > nword_sections_th:
                        tag_blocks[-1] = section_heading

                if (not reached_end
                    and filtered_text
                    and (max(sizes, default=0) >= size_threshold * size_mode
                         or nwords > 50)):
                    if section_to_find == "ABSTRACT" and nwords > nword_abstract_th:
                        tag = "ABSTRACT"
                        if word_list[-1][-1] == ".":
                            section_to_find = "INTRODUCTION"
                    elif section_to_find == "INTRODUCTION":
                        if nwords > nword_sections_th:
                            tag = "INTRODUCTION"
                            section_heading, section_to_find = (
                                "INTRODUCTION", "NEXTHEADING")
                        else:
                            tag = "UNDEFINED"
                    elif (section_to_find == "NEXTHEADING" and nwords > nword_sections_th):
                        tag = section_heading

                    text_blocks.append(filtered_text)
                    tag_blocks.append(tag)
                    nwords_in_blocks.append(nwords)
    else:
        logger.warning(f"PDF {pdf_path} likely corrupted")

    sections["pdf_abstract"] = "\n".join(
        text
        for t, text in zip(tag_blocks, text_blocks)
        if any(s in t.lower() for s in {"abstract"}))

    sections["pdf_introduction"] = "\n".join(
        text
        for t, text in zip(tag_blocks, text_blocks)
        if any(s in t.lower() for s in {"introduction", "related work", "i.", "ii."}))

    sections["pdf_results"] = "\n".join(
        text
        for t, text in zip(tag_blocks, text_blocks)
        if any(s in t.lower() for s in {"results", "experiment", "i.", "ii."}))

    sections["pdf_discussion"] = "\n".join(
        text
        for t, text in zip(tag_blocks, text_blocks)
        if any(s in t.lower() for s in {"discussion", "conclusion", "v.", "vi."}))

    sections["pdf_methods"] = "\n".join(
        text
        for t, text in zip(tag_blocks, text_blocks)
        if any(
            s in t.lower()
            for s in {
                "methods",
                "materials",
                "experimental",
                "materials and methods",
                "experimental procedure"}))

    sections["pdf_text"] = "\n".join(
        text
        for t, text in zip(tag_blocks, text_blocks)
        if not any(s in t.lower() for s in {"undefined"}))

    sections["author_list"] = authorlist

    return sections


def extract_pdf_sections(
        pdf_path: str,
        engine: str = "pdfminer",
        authorlist: str = None,
        valid_pages: List[int] = None,
        possible_section_headings: Set = None,
        size_threshold: float = None) -> dict:

    if "pdfminer" in engine.lower():
        return _extract_sections_pdfminer(
            pdf_path, authorlist, valid_pages, possible_section_headings, size_threshold)
    else:
        return None
