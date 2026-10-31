"""`moonshot` command-line interface."""

from typing import Annotated

import typer

from crypto_moonshot import __version__

app = typer.Typer(no_args_is_help=True, help="Crypto research and trading toolkit for Binance.")


def _print_version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_print_version, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """Crypto research and trading toolkit for Binance."""
