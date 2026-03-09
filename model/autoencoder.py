from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as weight_init

from utils.ae_utils import MSEloss

_ACTIVATIONS = {
    "selu": F.selu,
    "relu": F.relu,
    "relu6": F.relu6,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "elu": F.elu,
    "lrelu": F.leaky_relu,
    "swish": lambda x: x * torch.sigmoid(x),
    "none": lambda x: x,
}


def activation(input: torch.Tensor, kind: str) -> torch.Tensor:
    try:
        return _ACTIVATIONS[kind](input)
    except KeyError:
        raise ValueError(f"Unknown non-linearity type: {kind}")


class CFAutoEncoder(nn.Module):

    def __init__(
            self,
            layer_sizes: List[int],
            nl_type: str = "selu",
            is_constrained: bool = True,
            learn_rate: float = 1e-4,
            dp_drop_prob: float = 0.0,
            last_layer_activations: bool = False,
    ):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._nl_type = nl_type
        self._dp_drop_prob = dp_drop_prob
        self._last_layer_activations = last_layer_activations
        self.is_constrained = is_constrained
        self._last = len(layer_sizes) - 2

        ######## ENCODE ########
        self.encoder = nn.ModuleList()
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            lin = nn.Linear(in_dim, out_dim)
            weight_init.xavier_uniform_(lin.weight)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)
            self.encoder.append(lin)

        # Dropout on code
        self.drop = nn.Dropout(p=self._dp_drop_prob) if self._dp_drop_prob > 0 else nn.Identity()

        ######## DECODE ########
        if is_constrained:
            # Tied weights: reuse encoder weights (transposed) in decode.
            # Need only separate biases.
            reversed_enc_layers = list(reversed(layer_sizes))
            self.decode_b = nn.ParameterList()
            for in_dim, out_dim in zip(reversed_enc_layers[:-1], reversed_enc_layers[1:]):
                # out_dim = next in reversed list, which matches original input side
                b = nn.Parameter(torch.zeros(out_dim))
                self.decode_b.append(b)
            self.decoder = None
        else:
            # Unconstrained decoder with its own Linear layers
            reversed_enc_layers = list(reversed(layer_sizes))
            self.decoder = nn.ModuleList()
            for in_dim, out_dim in zip(reversed_enc_layers[:-1], reversed_enc_layers[1:]):
                lin = nn.Linear(in_dim, out_dim)
                weight_init.xavier_uniform_(lin.weight)
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)
                self.decoder.append(lin)
            self.decode_b = None

        self.optimizer = optim.Adam(self.parameters(), lr=learn_rate)

    @property
    def code_dim(self) -> int:
        """Dimensionality of the latent representation (code)."""
        return self.encoder[-1].out_features

    def encode(self, x: torch.Tensor) -> torch.Tensor:

        for lin in self.encoder:
            x = activation(lin(x), self._nl_type)
        x = self.drop(x)
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        '''

        '''
        if self.is_constrained:
            # Tied weights: reuse encoder weights in reverse order
            out = z
            num_layers = len(self.encoder)
            for dec_idx in range(num_layers):
                enc_layer = self.encoder[num_layers - 1 - dec_idx]
                W = enc_layer.weight  # [out_dim, in_dim]
                b = self.decode_b[dec_idx]

                out = F.linear(out, W.t(), b)
                if dec_idx != self._last or self._last_layer_activations:
                    out = activation(out, self._nl_type)
            return out
        else:
            out = z
            for dec_idx, lin in enumerate(self.decoder):
                out = lin(out)
                if dec_idx != self._last or self._last_layer_activations:
                    out = activation(out, self._nl_type)
            return out

    def forward(self, batch, return_code: bool = False):
        """
        x : torch.Tensor
            Input [B, input_dim].
        return_code : bool, default False
            If True, returns (reconstruction, code) instead of just reconstruction.

        Returns
        -------
        recon : torch.Tensor
            Reconstructed input.
        code : torch.Tensor, optional
            Latent representation (if return_code=True).
        """
        code = self.encode(batch)
        recon = self.decode(code)
        if return_code:
            return recon, code
        return recon

    def calculate_loss(self, batch):

        batch = batch.to(self.device)

        preds = self(batch)

        loss, _ = MSEloss(preds, batch, size_average=True)

        return loss, batch.size(0)

    def predict_step(self, batch):
        """
        Adapter used by evaluate_model:
        takes a batch from the DataLoader and returns (y, y_hat)
        """
        batch = batch.to(self.device)
        mask = batch != 0

        # Forward pass
        y_hat = self(batch)

        y_true_flat = batch[mask]
        y_pred_flat = y_hat[mask]

        return y_true_flat, y_pred_flat
