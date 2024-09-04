import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from models.matrix_factorization import MatrixFactorization

class UserItemDataset(Dataset):
    def __init__(self, matrix):
        self.user_item_pairs = torch.nonzero(matrix, as_tuple=False)  # Only non-zero entries
        self.ratings = matrix[self.user_item_pairs[:, 0], self.user_item_pairs[:, 1]]

    def __len__(self):
        return len(self.user_item_pairs)

    def __getitem__(self, idx):
        user = self.user_item_pairs[idx, 0]
        item = self.user_item_pairs[idx, 1]
        rating = self.ratings[idx]
        return user, item, rating

def load_data():
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    dataset_path = "data/ml-latest-small.zip"
    if not os.path.exists(dataset_path):
        urllib.request.urlretrieve(url, dataset_path)
        with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
            zip_ref.extractall("data/")
    
    ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
    return ratings

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ratings = load_data()
    user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    user_item_matrix = torch.tensor(user_item_matrix.values, dtype=torch.float32).to(device)

    num_users, num_items = user_item_matrix.shape
    train_dataset = UserItemDataset(user_item_matrix)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

    num_factors = 20
    model = MatrixFactorization(num_users, num_items, num_factors).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-5)
    loss_function = torch.nn.MSELoss()

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            users, items, ratings = batch
            users = users.to(device)
            items = items.to(device)
            ratings = ratings.to(device)

            optimizer.zero_grad()
            predictions = model(users, items)
            loss = loss_function(predictions, ratings)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

if __name__ == "__main__":
    train_model()
