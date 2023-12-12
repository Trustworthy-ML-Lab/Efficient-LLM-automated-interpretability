from setuptools import setup

setup(
    name="neuron_explainer",
    packages=["neuron_explainer"],
    version="0.0.1",
    author="OpenAI",
    install_requires=[
        "httpx>=0.22",
        "scikit-learn",
        "boostedblob>=0.13.0",
        "tiktoken",
        "blobfile",
        "numpy",
        "pytest",
        "orjson",
        "pandas",
        "tenacity",
        "openai==0.28"
    ],
    url="",
    description="",
    python_requires='>=3.9',
)
