# Contributing

Bug reports, small fixes, model adapters and usability suggestions are welcome. Please follow the [code of conduct](CODE_OF_CONDUCT.md).

- [Ask questions or discuss ideas](https://github.com/EastonSell/model-looking-glass/discussions).
- [Report a bug or suggest a change](https://github.com/EastonSell/model-looking-glass/issues/new/choose).
- For a security concern, follow [SECURITY.md](SECURITY.md) instead of posting details publicly.
- For a larger change, open an issue first so we can agree on the scope.

## Send a change

1. Fork the repository and create a branch from `main`.
2. Keep the change focused. Add or update a regression test when behavior changes.
3. Open a pull request explaining what changed, why, and how you checked it. A draft is fine for work in progress.

Use Python 3.11 or newer. In a virtual environment, install the app and test dependencies:

```sh
python -m pip install -e ".[app,test]"
```

Run the tests nearest your change. For example, an installer change can start with:

```sh
python -m pytest tests/test_app_install.py -q
```

Record the commands and results in your PR, including anything you could not test. For UI changes, include a screenshot and check the [accessibility guidance](ACCESSIBILITY.md). For GPU changes, give the GPU, driver, PyTorch and CUDA versions; CPU results do not establish GPU behavior.

Keep generated captures, model weights, virtual environments and private prompts out of commits. Preserve saved-trace compatibility and keep dependency/security changes explicit. Don't hide failing checks or dismiss alerts to make a PR look ready.

## Find your way around

The [code map](looking_glass/README.md) points to the app, shared capture tools and historical studies. Start with the files for your change rather than reading the whole package. Keep existing import paths and study records intact unless a change includes their dependencies and reproducibility checks.
