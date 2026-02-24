import unittest

import torch
import torch.nn as nn

from ftta.tta import EATAMethod, SARMethod, TENTMethod, TTA_REGISTRY, _collect_tent_params


class _NormModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(8)
        self.head = nn.Linear(8, 1)

    def forward(self, x):
        return self.head(self.norm(x))


class _NoNormModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(8, 1)

    def forward(self, x):
        return self.head(x)


class TestTtaRegistry(unittest.TestCase):
    def test_tent_registered(self):
        model = _NoNormModel()
        method = TTA_REGISTRY.create(
            "tent",
            model=model,
            optimizer_type=torch.optim.SGD,
            tent_lr=1e-3,
            tent_steps=1,
            device="cpu",
        )
        self.assertIsInstance(method, TENTMethod)

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            TTA_REGISTRY.create("unknown_method")

    def test_sar_registered(self):
        model = _NoNormModel()
        method = TTA_REGISTRY.create(
            "sar",
            model=model,
            optimizer_type=torch.optim.SGD,
            sar_lr=1e-3,
            sar_steps=1,
            device="cpu",
        )
        self.assertIsInstance(method, SARMethod)

    def test_eata_registered(self):
        model = _NoNormModel()
        method = TTA_REGISTRY.create(
            "eata",
            model=model,
            optimizer_type=torch.optim.SGD,
            eata_lr=1e-3,
            eata_steps=1,
            device="cpu",
        )
        self.assertIsInstance(method, EATAMethod)

    def test_collect_tent_params_prefers_norm(self):
        model = _NormModel()
        params = _collect_tent_params(model)
        self.assertEqual(len(params), 2)
        self.assertTrue(all(p.requires_grad for p in params))

    def test_collect_tent_params_fallback_all(self):
        model = _NoNormModel()
        params = _collect_tent_params(model)
        all_params = [p for p in model.parameters() if p.requires_grad]
        self.assertEqual(len(params), len(all_params))

    def test_tent_adapt_predict_shape(self):
        model = _NoNormModel()
        tent = TENTMethod(
            model=model,
            optimizer_type=torch.optim.SGD,
            tent_lr=1e-3,
            tent_steps=1,
            device="cpu",
        )
        x = torch.randn(5, 8)
        y = tent.adapt_predict(x)
        self.assertEqual(tuple(y.shape), (5,))

    def test_sar_adapt_predict_shape(self):
        model = _NoNormModel()
        sar = SARMethod(
            model=model,
            optimizer_type=torch.optim.SGD,
            sar_lr=1e-3,
            sar_steps=1,
            device="cpu",
        )
        x = torch.randn(5, 8)
        y = sar.adapt_predict(x)
        self.assertEqual(tuple(y.shape), (5,))

    def test_eata_adapt_predict_shape(self):
        model = _NoNormModel()
        eata = EATAMethod(
            model=model,
            optimizer_type=torch.optim.SGD,
            eata_lr=1e-3,
            eata_steps=1,
            device="cpu",
        )
        x = torch.randn(5, 8)
        y = eata.adapt_predict(x)
        self.assertEqual(tuple(y.shape), (5,))


if __name__ == "__main__":
    unittest.main()
