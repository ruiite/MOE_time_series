import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import copy


###############################
# 1. Expert: Transformer Decoder Block for MoE
###############################
class ExpertTransformerDecoder(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float = 0.1
    ):
        """
        A mini transformer decoder block used as an expert.
        d_model: input and output dimension.
        n_heads: number of attention heads for self-attention.
        dim_feedforward: hidden dimension of the internal feed-forward network.
        dropout: dropout probability.
        """
        super(ExpertTransformerDecoder, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        # Use GELU instead of ReLU for smoother activation
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
            nn.GELU(),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention sub-layer
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        # Feed-forward sub-layer
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)
        return x


###############################
# 2. Gating Network and MoE Module (using Transformer Decoder Experts)
###############################
class GatingNetwork(nn.Module):
    def __init__(self, input_dim: int, num_experts: int):
        """
        Simple gating network that uses a linear projection and softmax to create a
        probability distribution over experts.
        """
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.fc(x)
        probs = F.softmax(logits, dim=-1)
        return probs


class SelfAttentionGatingNetwork(nn.Module):
    def __init__(self, d_model: int, num_experts: int):
        """
        Uses self-attention over the token representations, then projects the result to gating logits.
        d_model: input (and output) dimension.
        num_experts: number of experts to gate over.
        """
        super(SelfAttentionGatingNetwork, self).__init__()
        # Use a single-head self-attention layer for simplicity (can be extended to multi-head).
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=1, batch_first=True
        )
        # Use a linear layer to map the self-attended representation to gating logits per expert.
        self.fc = nn.Linear(d_model, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (batch, seq_len, d_model).
        Returns:
           gating_probs: Tensor of shape (batch, seq_len, num_experts) after applying softmax.
        """
        # Apply self-attention: here, queries, keys, and values are all x.
        # The self-attention output will be a context-enhanced version of x.
        attn_output, _ = self.self_attn(x, x, x)
        # Project the self-attended output to get gating logits for each expert.
        gating_logits = self.fc(attn_output)
        # Apply softmax along the expert dimension to obtain a probability distribution.
        gating_probs = F.softmax(gating_logits, dim=-1)
        return gating_probs


class AttentionGatingNetwork(nn.Module):
    def __init__(self, d_model: int, num_experts: int):
        super(AttentionGatingNetwork, self).__init__()
        # Learnable expert query vectors
        # self.expert_queries = nn.Linear(d_model, num_experts)
        self.expert_queries = nn.Parameter(torch.randn(num_experts, d_model))

        # Optionally, you can include a linear projection for keys
        self.key_proj = nn.Linear(d_model, d_model)
        self.scale = d_model**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model)
        # Project x to keys
        keys = self.key_proj(x)  # (batch, seq_len, d_model)
        # Compute attention scores between each token and expert queries.
        # expert_queries shape: (num_experts, d_model) -> (1, 1, num_experts, d_model)
        # expert_q = self.expert_queries(x)
        expert_q = self.expert_queries.unsqueeze(0).unsqueeze(0)

        keys = keys.unsqueeze(2)
        # keys shape: (batch, seq_len, 1, d_model)
        # Compute scaled dot-product attention scores
        attn_scores = (keys * expert_q).sum(
            -1
        ) * self.scale  # (batch, seq_len, num_experts)
        # Softmax to get gating probabilities over experts
        gating_probs = F.softmax(attn_scores, dim=-1)
        return gating_probs


class SimplifiedGatingNetwork(nn.Module):
    def __init__(self, d_model: int, num_experts: int):
        super(SimplifiedGatingNetwork, self).__init__()
        # Learnable expert query vectors
        # self.expert_queries = nn.Linear(d_model, num_experts)
        self.expert_queries = nn.Parameter(torch.randn(num_experts))

        # Optionally, you can include a linear projection for keys
        self.key_proj = nn.Linear(d_model, num_experts)
        self.scale = d_model**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model)
        # Project x to keys
        keys = self.key_proj(x)  # (batch, seq_len, d_model)
        # Compute attention scores between each token and expert queries.
        # expert_queries shape: (num_experts, d_model) -> (1, 1, num_experts, d_model)
        # expert_q = self.expert_queries(x)
        expert_q = self.expert_queries.unsqueeze(0)
        # expert_q = self.expert_queries.unsqueeze(0).unsqueeze(0)

        # keys shape: (batch, seq_len, 1, d_model)
        # Compute scaled dot-product attention scores
        attn_scores = torch.matmul(keys, expert_q)
        # attn_scores = (keys * expert_q).sum(
        #     -1
        # ) * self.scale  # (batch, seq_len, num_experts)
        # Softmax to get gating probabilities over experts
        gating_probs = F.softmax(attn_scores, dim=-1)
        return gating_probs


class NewAttentionGatingNetwork(nn.Module):
    def __init__(self, d_model: int, num_experts: int):
        super(AttentionGatingNetwork, self).__init__()
        # Learnable expert query vectors
        # self.expert_queries = nn.Parameter(torch.randn(num_experts, d_model))
        # Optionally, you can include a linear projection for keys
        self.key_proj = nn.Linear(d_model, num_experts)
        self.queue_proj = nn.Linear(d_model, num_experts)
        self.value_proj = nn.Linear(d_model, num_experts)
        self.scale = d_model**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model)
        # Project x to keys
        keys = self.key_proj(x)  # (batch, seq_len, d_model)
        queues = self.queue_proj(x)
        values = self.value_proj(x)

        scores = torch.matmul(queues, keys.T)

        scaled_scores = scores / self.scale

        attention_weights = F.softmax(scaled_scores, dim=-1)
        output = torch.matmul(attention_weights, values)

        output_softmax = F.softmax(output)
        return output_softmax


class AttentionMoEGating(nn.Module):
    def __init__(self, input_dim: int, model_dim: int, num_experts: int):
        """
        Gating mechanism for a neural network mixture-of-experts using an attention-like approach.
        It is inspired by the DummyInputHMMTransitions algorithm but computes the gating
        weights via dot-product attention.

        Args:
            input_dim: Dimension of the raw input features.
            model_dim: Dimension to which inputs are projected for computing attention.
            num_experts: Number of experts (discrete states).
        """
        super(AttentionMoEGating, self).__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.model_dim = model_dim

        # Linear projection: projects raw input (input_dim) to model_dim.
        self.W_input = nn.Linear(input_dim, model_dim, bias=False)

        # Learned expert key embeddings (one per expert); shape: (num_experts, model_dim)
        self.expert_keys = nn.Parameter(torch.randn(num_experts, model_dim))

        # Learned fixed "markov" contribution (simulating the part from the previous state).
        # This is of shape (num_experts,).
        self.markov_contrib = nn.Parameter(torch.randn(num_experts))

        # Additional bias term (per expert); shape: (num_experts,).

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Given an array of covariates X of shape (T, input_dim), compute gating (transition)
        probabilities for each time step using a dot-product attention mechanism.

        Returns a tensor of shape (T, num_experts), where each row is a probability distribution
        (i.e. the entries sum to 1) over experts.
        """
        # Project the input covariates: shape goes from (T, input_dim) to (T, model_dim)
        X_proj = self.W_input(X)

        # Compute the dot-product scores between inputs and expert keys:
        # expert_keys.t() has shape (model_dim, num_experts), so the result is (T, num_experts).
        # Scale the scores by sqrt(model_dim) (as in standard attention mechanisms).
        scale = self.model_dim**0.5
        attention_scores = torch.matmul(X_proj, self.expert_keys.t()) / scale

        # Combine the attention scores with the fixed "markov" contribution and bias.
        # Both self.markov_contrib and self.bias have shape (num_experts,), so we unsqueeze them to add along the time dimension.
        psi = (
            attention_scores
            + self.markov_contrib.unsqueeze(0)  # + self.bias.unsqueeze(0)
        )

        # Apply softmax over the expert dimension (last dimension) to produce a probability distribution.
        gating_weights = F.softmax(psi, dim=-1)
        return gating_weights


class MoETransformerDecoderFFN(nn.Module):
    def __init__(
        self,
        model_dim: int,
        ffn_dim: int,
        num_experts: int,
        top_k: int,
        expert_n_heads: int,
        dropout: float = 0.1,
        gating: str = "attention",
    ):
        """
        model_dim: input/output dimension.
        ffn_dim: hidden dimension for each expert.
        num_experts: total experts.
        top_k: number of experts selected per token.
        expert_n_heads: number of heads for the transformer-decoder experts.
        dropout: dropout probability.
        """
        super(MoETransformerDecoderFFN, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # Initialize transformer-decoder experts
        self.experts = nn.ModuleList(
            [
                ExpertTransformerDecoder(model_dim, expert_n_heads, ffn_dim, dropout)
                for _ in range(num_experts)
            ]
        )
        if gating == "attention":
            self.gating = AttentionGatingNetwork(model_dim, num_experts)
        elif gating == "classic":
            self.gating = GatingNetwork(model_dim, num_experts)
        elif gating == "paper":
            self.gating = AttentionMoEGating(model_dim, ffn_dim, num_experts)
        elif gating == "new":
            self.gating = NewAttentionGatingNetwork(model_dim, num_experts)
        elif gating == "simple":
            self.gating = SimplifiedGatingNetwork(model_dim, num_experts)

        # self.gating = SelfAttentionGatingNetwork(model_dim, num_experts)

    def forward(self, x: torch.Tensor, return_gate: bool = False) -> torch.Tensor:
        # x: (batch, seq_len, model_dim)
        batch_size, seq_len, model_dim = x.size()
        gating_probs = self.gating(x)  # (batch, seq_len, num_experts)

        # Top-k selection: if top_k < num_experts, mask out lower scoring experts
        if self.top_k < self.num_experts:
            topk_vals, topk_idx = torch.topk(gating_probs, self.top_k, dim=-1)
            mask = torch.zeros_like(gating_probs)
            mask = mask.scatter(dim=-1, index=topk_idx, value=1.0)
            gated_probs = gating_probs * mask
            # Explicit normalization: divide by the sum over experts to ensure proper weighting
            gated_probs = gated_probs / (gated_probs.sum(dim=-1, keepdim=True) + 1e-9)
        else:
            gated_probs = gating_probs

        # Compute outputs from all experts
        expert_outputs = [
            expert(x) for expert in self.experts
        ]  # each: (batch, seq_len, model_dim)
        expert_outputs = torch.stack(
            expert_outputs, dim=-1
        )  # (batch, seq_len, model_dim, num_experts)

        # Multiply by gating probabilities and sum over experts
        gated_probs_expanded = gated_probs.unsqueeze(
            -2
        )  # (batch, seq_len, 1, num_experts)
        moe_output = (expert_outputs * gated_probs_expanded).sum(
            dim=-1
        )  # (batch, seq_len, model_dim)
        if return_gate:
            # return (output, gating_probs, topk_idx)
            return moe_output, gating_probs, topk_idx
        return moe_output


###############################
# 3. Transformer Layer Using MoE with Decoder Experts
###############################
class MoETransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        num_experts: int,
        top_k: int,
        expert_n_heads: int,
        dropout: float = 0.1,
        gating: str = "attention",
    ):
        """
        A Transformer layer that applies self-attention followed by the MoE module.
        """
        super(MoETransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.moe_decoder = MoETransformerDecoderFFN(
            model_dim=d_model,
            ffn_dim=ffn_dim,
            num_experts=num_experts,
            top_k=top_k,
            expert_n_heads=expert_n_heads,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, return_gate=False) -> torch.Tensor:
        attn_output, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_output))
        if return_gate:
            out, gating, topk = self.moe_decoder(x, return_gate=True)
            return self.norm2(x + self.dropout(out)), gating, topk
        else:
            decoder_output = self.moe_decoder(x)
            x = self.norm2(x + self.dropout(decoder_output))
            return x


###############################
# 4. Full Forecasting Model with MoE
###############################
class MoETransformerForecaster(nn.Module):
    def __init__(
        self,
        input_size: int = 24,
        target_size: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 128,
        num_experts: int = 4,
        top_k: int = 2,
        use_decoder_expert: bool = True,
        expert_n_heads: int = None,
        dropout: float = 0.1,
        gating: str = "attention",
    ):
        """
        input_size: input window length (default 24).
        target_size: forecast horizon (default 1 for one-step ahead).
        d_model: model embedding dimension.
        n_heads: self-attention heads.
        num_layers: number of Transformer layers.
        ffn_dim: hidden dimension for expert feed-forward networks.
        num_experts: number of MoE experts.
        top_k: how many experts to activate per token.
        use_decoder_expert: if True, uses transformer-decoder experts.
        expert_n_heads: number of attention heads for each expert (defaults to n_heads if None).
        dropout: dropout probability.
        """
        super(MoETransformerForecaster, self).__init__()
        self.input_size = input_size
        self.target_size = target_size
        self.d_model = d_model

        # Project the univariate input to d_model dimensions.
        self.value_embed = nn.Linear(1, d_model)
        # Learnable positional embeddings.
        self.pos_embedding = nn.Parameter(torch.zeros(1, input_size, d_model))

        self.name = "MOE" + gating

        expert_n_heads = expert_n_heads if expert_n_heads is not None else n_heads
        if use_decoder_expert:
            self.layers = nn.ModuleList(
                [
                    MoETransformerDecoderLayer(
                        d_model,
                        n_heads,
                        ffn_dim,
                        num_experts,
                        top_k,
                        expert_n_heads,
                        dropout,
                        gating,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "Only transformer-decoder experts are implemented in this example."
            )

        # Output layer to generate forecasts.
        self.output_layer = nn.Linear(d_model, target_size)

    def forward(self, x: torch.Tensor, return_gate: bool = False) -> torch.Tensor:
        # x: shape (batch, input_size) or (batch, input_size, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        batch_size, seq_len, _ = x.size()
        if seq_len > self.input_size:
            raise ValueError(
                f"Input length {seq_len} exceeds model input_size {self.input_size}."
            )

        # Embed the input and add positional encoding.
        x_embed = self.value_embed(x)
        x_embed = x_embed + self.pos_embedding[:, :seq_len, :]

        out = x_embed
        all_gates, all_topk = [], []
        for layer in self.layers:
            if return_gate:
                out, gating, topk = layer(out, return_gate=True)
                all_gates.append(gating)  # list of (batch, seq_len, num_experts)
                all_topk.append(topk)  # list of (batch, seq_len, top_k)
            else:
                out = layer(out)

        final = out[:, -1, :]
        forecast = self.output_layer(final).squeeze(-1)

        if return_gate:
            return forecast, all_gates, all_topk
        return forecast


def count_active_params(model: torch.nn.Module, top_k: int) -> int:
    """
    Estimate the number of parameters actively used during inference when only top_k experts per MoE module are computed.

    The function computes:
      active_params = non-MoE params + sum_over_MoE_modules[(top_k / num_experts_module) * expert_params_module]

    Args:
        model (nn.Module): The forecasting model.
        top_k (int): The number of experts activated per token.

    Returns:
        int: Estimated active parameter count.
    """
    # Total parameters in the model (all are always stored).
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    moe_expert_params_total = 0  # Sum of parameters for all MoE experts.
    active_expert_params = (
        0  # Sum of parameters from experts that would be computed in a forward pass.
    )

    # Iterate over all modules to find MoE modules.
    for module in model.modules():
        if isinstance(module, MoETransformerDecoderFFN):
            num_experts = len(module.experts)
            # Count parameters for experts in this MoE module.
            expert_params_module = sum(
                sum(p.numel() for p in expert.parameters() if p.requires_grad)
                for expert in module.experts
            )
            moe_expert_params_total += expert_params_module
            # Only top_k experts are used at inference.
            active_expert_params += (top_k / num_experts) * expert_params_module

    # All other parameters (e.g. input projection, positional embeddings, self-attention outside MoE, gating network, etc.)
    non_moe_params = total_params - moe_expert_params_total

    # The total "active" parameters is the non-MoE parameters plus the fraction of MoE expert parameters.
    active_total = non_moe_params + int(active_expert_params)
    return active_total


###############################
# 5. Example Training on Synthetic Data
###############################
if __name__ == "__main__":
    import numpy as np
    import torch.optim as optim
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from matplotlib import pyplot as plt
    from utils import create_dataset, get_dataset

    # Generate synthetic univariate time series data (e.g. noisy sine wave)
    # np.random.seed(0)
    # time = np.arange(0, 100, 0.1)
    # series = np.sin(0.2 * time) + 0.1 * np.random.randn(len(time))
    device = "cpu"
    time_series = get_dataset("solar")
    scaler = MinMaxScaler()
    series = scaler.fit_transform(time_series.reshape(-1, 1)).flatten()

    # Prepare training data using a sliding window of 24 time steps.
    window_size = 64
    horizon = 48

    X_train, y_train, X_val, y_val = create_dataset(
        window_size=window_size, horizon=horizon, series=series
    )

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    # Instantiate the model.
    model = MoETransformerForecaster(
        input_size=window_size,
        target_size=horizon,
        d_model=48,
        n_heads=4,
        num_layers=1,
        ffn_dim=64,
        num_experts=8,
        top_k=2,
        use_decoder_expert=True,  # use transformer decoder experts
        expert_n_heads=2,  # number of heads for the expert blocks
        dropout=0.1,
        gating="attention",
    )

    # print(count_active_params(model, 2))
    model.to(device)
    # Define loss and optimizer.
    criterion = nn.MSELoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    best_val_mse = float("inf")
    best_model_state = None

    # Simple training loop.
    for epoch in range(20):
        model.train()
        permutation = torch.randperm(X_train.size(0))
        batch_size = 64
        epoch_loss = 0.0
        for i in range(0, X_train.size(0), batch_size):
            idx = permutation[i : i + batch_size]
            batch_x = X_train[idx]
            batch_y = y_train[idx]
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        epoch_loss /= X_train.size(0)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred, gates, chosen = model(X_val, True)
            chosen = np.asarray(chosen).squeeze(0)

            all_pairs = [
                tuple(sorted(pair.tolist()))
                for sample in chosen  # sample: shape (64,2)
                for pair in sample  # pair:   shape (2,)
            ]

            # 3. Count them
            pair_counts = Counter(all_pairs)

            # 4. See your dict
            print(pair_counts)
            val_loss = criterion(val_pred, y_val).item()
        print(f"Epoch {epoch + 1}: Train MSE={epoch_loss:.4f}, Val MSE={val_loss:.4f}")

        if val_loss < best_val_mse:
            best_val_mse = val_loss
            best_model_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        val_pred, gates, chosen = model(X_val, True)
        val_loss = criterion(val_pred, y_val).item()

    result_dict = {}
    result_dict["val_pred"] = val_pred
    result_dict["gates"] = gates
    result_dict["chosen"] = chosen
    result_dict["y_val"] = y_val

    import pickle

    with open("results.pkl", "wb") as f:
        pickle.dump(result_dict, f)

    def calculate_metrics(y_true, y_pred):
        # Denormalize the values
        # y_true = scaler.inverse_transform(y_true.detach().numpy())
        # y_pred = scaler.inverse_transform(y_pred.detach().numpy())

        y_true = y_true.detach().numpy()
        y_pred = y_pred.detach().numpy()

        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        smape = 100 * np.mean(
            2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))
        )

        # MASE calculation
        naive_forecast = np.roll(y_true, 1)  # Shifted series for naive forecast
        naive_error = np.mean(np.abs(y_true[1:] - naive_forecast[1:]))
        mase = mae / naive_error

        return {"MAE": mae, "MSE": mse, "RMSE": rmse, "SMAPE": smape, "MASE": mase}

    print(calculate_metrics(y_val, val_pred))

    plt.plot(val_pred, color="black")
    plt.plot(y_val, color="green")
    plt.show()
