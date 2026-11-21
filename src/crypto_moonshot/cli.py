"""`moonshot` command-line interface."""

from typing import Annotated

import typer

from crypto_moonshot import __version__
from crypto_moonshot.backtest_cli import backtest_app
from crypto_moonshot.data_cli import data_app
from crypto_moonshot.exchange_cli import exchange_app
from crypto_moonshot.live_cli import live_app, provider_app

app = typer.Typer(no_args_is_help=True, help="Crypto research and trading toolkit for Binance.")

app.add_typer(data_app, name="data")
app.add_typer(backtest_app, name="backtest")
app.add_typer(exchange_app, name="exchange")
app.add_typer(live_app, name="live")
app.add_typer(provider_app, name="provider")


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
