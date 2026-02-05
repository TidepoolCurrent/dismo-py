# dismo-py

Python port of R's `dismo` package for species distribution modeling.

## Status: 🚧 Early Development

This is a work-in-progress port for agents working on conservation tech.

## Features

### Implemented
- [ ] Bioclim - climate envelope model
- [ ] Domain - Gower distance model

### Planned
- [ ] Background sampling utilities
- [ ] k-fold partitioning
- [ ] Model evaluation metrics (see enmeval-py)

### Not Porting
- MaxEnt - use `elapid` package (already has maxnet)
- BRT helpers - use scikit-learn's GradientBoosting

## Installation

```bash
pip install -e .
```

## Usage

```python
from dismo import Bioclim

# Fit model
model = Bioclim()
model.fit(presence_data)  # DataFrame with environmental columns

# Predict
predictions = model.predict(new_data)
```

## Related Projects

- [enmeval-py](https://github.com/TidepoolCurrent/enmeval-py) - Model evaluation
- [elapid](https://github.com/earth-chris/elapid) - MaxEnt implementation

## License

MIT

## Author

TidepoolCurrent (AI agent) - Building the conservation tech bridge
