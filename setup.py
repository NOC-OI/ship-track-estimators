from setuptools import setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

from src.track_estimators.version import __version__

setup(
    name="track_estimators",
    version=__version__,
    packages=["track_estimators"],
    package_dir={"track_estimators": "src/track_estimators"},
    install_requires=requirements,
    url="https://github.com/NOC-OI/ship-track-estimators",
    license="GPL",
    author="Joao Morado",
    author_email="joao.morado@noc.ac.uk",
    description="A package for estimation of historical ship tracks using Kalman filters.",
)
