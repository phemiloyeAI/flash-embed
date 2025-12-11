from pathlib import Path
from setuptools import setup, find_packages


def read_requirements() -> list[str]:
    req_path = Path(__file__).parent / "requirements.txt"
    if req_path.is_file():
        return [line.strip() for line in req_path.read_text().splitlines() if line.strip() and not line.startswith("#")]
    return []


setup(
    name="flash-embed",
    version="0.0.1",
    description="High-throughput image embedding pipeline",
    author="Femiloye Oyerinde, Tomiwa Samuel",
    license="MIT",
    packages=find_packages(),
    install_requires=read_requirements(),
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "flash-embed=flash_embed.cli:main",
        ]
    },
)
