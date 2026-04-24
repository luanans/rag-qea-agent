import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

PAPERS: dict[str, dict[str, str]] = {
    "attention": {
        "arxiv_id": "1706.03762",
        "title": "Attention Is All You Need",
        "url": "https://arxiv.org/pdf/1706.03762.pdf",
    },
    "bert": {
        "arxiv_id": "1810.04805",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "url": "https://arxiv.org/pdf/1810.04805.pdf",
    },
    "rag": {
        "arxiv_id": "2005.11401",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "url": "https://arxiv.org/pdf/2005.11401.pdf",
    },
}


def download_papers(output_dir: str = "./data/papers") -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    downloaded: dict[str, Path] = {}

    for paper_id, info in PAPERS.items():
        dest = output_path / f"{paper_id}.pdf"

        if dest.exists():
            logger.info(
                "Paper '%s' already exists at %s. Skipping download.", paper_id, dest
            )
            downloaded[paper_id] = dest
            continue

        logger.info("Downloading '%s' from %s ...", info["title"], info["url"])
        try:
            response = requests.get(info["url"], timeout=60, stream=True)
            response.raise_for_status()

            with open(dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(
                "Saved '%s' → %s (%d KB)", paper_id, dest, dest.stat().st_size // 1024
            )
            downloaded[paper_id] = dest

        except requests.RequestException as exc:
            logger.error("Failed to download '%s': %s", paper_id, exc)
            raise

    return downloaded
