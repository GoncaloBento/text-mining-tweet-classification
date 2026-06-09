from src.transformer_trainer import run_trainer as _run


def run_trainer(n_samples=500, notes=""):
    return _run("deberta", n_samples=n_samples, notes=notes)
