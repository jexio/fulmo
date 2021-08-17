import torch
import torch.nn as nn


class OnlineLabelSmoothing(nn.Module):
    """Implements Online Label Smoothing from paper https://arxiv.org/pdf/2011.12562.pdf."""

    def __init__(self, n_classes: int, alpha: float, smoothing: float = 0.1) -> None:
        """Create a new instance of `OnlineLabelSmoothing`.

        Args:
            n_classes: Number of classes of the classification problem
            alpha: Term for balancing soft_loss and hard_loss
            smoothing: Smoothing factor to be used during first epoch in soft_loss

        Raises:
            ValueError: if `alpha` or `smoothing` greater than 1 or less than 0
        """
        super(OnlineLabelSmoothing, self).__init__()
        if 0.0 > alpha > 1.0:
            raise ValueError("alpha must be in range [0, 1].")
        if 0.0 > smoothing > 1.0:
            raise ValueError("smoothing must be in range [0, 1].")

        self._alpha = alpha
        self._n_classes = n_classes

        # With alpha / (n_classes - 1) ----> Alternative
        self.register_buffer("supervise", torch.zeros(n_classes, n_classes))
        self.supervise.fill_(smoothing / (n_classes - 1))
        self.supervise.fill_diagonal_(1 - smoothing)

        # Update matrix is used to supervise next epoch
        self.register_buffer("update", torch.zeros_like(self.supervise))
        # For normalizing we need a count for each class
        self.register_buffer("idx_count", torch.zeros(n_classes))
        self.hard_loss = nn.CrossEntropyLoss()

    def forward(self, outputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the final loss."""
        soft_loss = self.soft_loss(outputs, target)
        hard_loss = self.hard_loss(outputs, target)
        return self._alpha * hard_loss + (1 - self._alpha) * soft_loss

    def soft_loss(self, outputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculates the soft loss and calls step to update `update`.

        Args:
            outputs: Predicted logits.
            target: Ground truth labels.

        Returns:
            Calculates the soft loss based on current supervise matrix.
        """
        outputs = outputs.log_softmax(dim=-1)
        with torch.no_grad():
            self.step(outputs.exp(), target)
            true_dist = torch.index_select(self.supervise, 1, target).swapaxes(-1, -2)
        return torch.mean(torch.sum(-true_dist * outputs, dim=-1))

    def step(self, outputs: torch.Tensor, target: torch.Tensor) -> None:
        """Updates `update` with the probabilities of the correct predictions and updates `idx_count` counter.

        Steps:
            1. Calculate correct classified examples.
            2. Filter `outputs` based on the correct classified.
            3. Add `y_h_f` rows to the `j` (based on y_h_idx) column of `memory`.
            4. Keep count of # samples added for each `y_h_idx` column.
            5. Average memory by dividing column-wise by result of step (4).
        Note on (5): This is done outside this function since we only need to
                     normalize at the end of the epoch.

        Args:
            outputs: Predicted logits.
            target: Ground truth labels.
        """
        # 1. Calculate predicted classes
        y_h_idx = outputs.argmax(dim=-1)
        # 2. Filter only correct
        mask = torch.eq(y_h_idx, target)
        y_h_c = outputs[mask]
        y_h_idx_c = y_h_idx[mask]
        # 3. Add y_h probabilities rows as columns to `memory`
        self.update.index_add_(1, y_h_idx_c, y_h_c.swapaxes(-1, -2))
        # 4. Update `idx_count`
        self.idx_count.index_add_(0, y_h_idx_c, torch.ones_like(y_h_idx_c, dtype=torch.float32))

    def next_epoch(self) -> None:
        """This function should be called at the end of the epoch."""
        # 5. Divide memory by `idx_count` to obtain average (column-wise)
        self.idx_count[torch.eq(self.idx_count, 0)] = 1  # Avoid 0 denominator
        # Normalize by taking the average
        self.update /= self.idx_count
        self.idx_count.zero_()
        self.supervise = self.update
        self.update = self.update.clone().zero_()


__all__ = ["OnlineLabelSmoothing"]
