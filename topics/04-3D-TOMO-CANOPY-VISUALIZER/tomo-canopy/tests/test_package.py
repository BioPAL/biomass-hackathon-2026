"""Package smoke tests."""

import tomo_canopy
from tomo_canopy import cubegen, methods, processors, transforms, visualisation
from tomo_canopy.providers import als, gedi, tls, tomograms


def test_public_modules_are_importable() -> None:
    """The initial public package structure can be imported."""
    assert tomo_canopy.__version__
    assert all(
        module.__name__.startswith("tomo_canopy")
        for module in (
            als,
            cubegen,
            gedi,
            methods,
            processors,
            tls,
            tomograms,
            transforms,
            visualisation,
        )
    )
