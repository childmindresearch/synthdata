"""Shared plotting helpers: figure saving for both matplotlib and Plotly figures."""

from pathlib import Path

from synthdata.utils import ensure_dir, get_logger

logger = get_logger(__name__)


def save_matplotlib_figure(fig, path: str | Path, dpi: int = 150, formats=("png",)) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        logger.info("Saved figure: %s", out)


def save_plotly_figure(fig, path: str | Path, formats=("html",)) -> None:
    """Save a Plotly figure. Falls back to HTML if 'png'/'pdf' requested but
    kaleido (the static-image exporter) isn't installed."""
    path = Path(path)
    ensure_dir(path.parent)
    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        try:
            if fmt == "html":
                fig.write_html(str(out))
            else:
                fig.write_image(str(out))
            logger.info("Saved figure: %s", out)
        except (ValueError, ImportError) as exc:
            logger.warning(
                "Could not save %s (%s); falling back to HTML. Install 'kaleido' for "
                "static image export of Plotly figures.",
                out,
                exc,
            )
            fallback = path.with_suffix(".html")
            fig.write_html(str(fallback))
            logger.info("Saved figure: %s", fallback)
