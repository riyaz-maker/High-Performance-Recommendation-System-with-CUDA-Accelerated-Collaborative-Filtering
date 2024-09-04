import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, num_factors, dropout_rate=0.2):
        super(MatrixFactorization, self).__init__()
        self.user_factors = nn.Embedding(num_users, num_factors)
        self.item_factors = nn.Embedding(num_items, num_factors)
        self.dropout = nn.Dropout(dropout_rate)

        # Initialize embeddings with small random values
        nn.init.normal_(self.user_factors.weight, std=0.01)
        nn.init.normal_(self.item_factors.weight, std=0.01)

    def forward(self, user, item):
        user_embedding = self.dropout(self.user_factors(user))
        item_embedding = self.dropout(self.item_factors(item))
        return (user_embedding * item_embedding).sum(1)
