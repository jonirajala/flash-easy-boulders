import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import utility functions from a separate module
from utils import load_data_from_db, preprocess_data, get_features, RoutesDataset, SimpleMLPRegression, initialize_database_if_needed

# Constants
BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
MIN_ASCENTS = 5 # needs this many ascents to boulder to be included

def load_and_preprocess_data():
    """Load and preprocess data."""
    print("Loading data")
    holes_df, routes_df, routes_grade_df, _ = load_data_from_db()
    print("Data Loaded")

    print("Preprocessing data")
    routes_l1 = preprocess_data(routes_df, routes_grade_df, min_ascents=MIN_ASCENTS)
    print(f"Data preprocessed, {len(routes_l1)} boulder problems")

    print("Extracting features")
    routes = get_features(routes_l1, holes_df)
    print(f"Features extracted. Got {routes.shape} features")

    return routes, routes_l1

def create_datasets_and_loaders(routes, routes_l1):
    """Create datasets and dataloaders."""
    print("Creating dataset")
    full_dataset = RoutesDataset(routes, routes_l1['difficulty_average'].values, routes_l1['angle'].values)

    # Splitting the dataset
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    print("Creating dataloaders")
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader

def train_and_evaluate_model(train_loader, test_loader):
    """Train and evaluate the model."""
        
    data_iterator = iter(train_loader)
    images, _ = next(data_iterator)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleMLPRegression(images.shape[-1]).to(device).double()  # Convert model to double precision

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    test_losses, train_losses = [], []

    print("Starting training")
    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch, device)
        test_loss = evaluate(model, test_loader, criterion, device)

        test_losses.append(test_loss)
        train_losses.append(train_loss)

        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    save_model(model, "model.pth")
    plot_losses(train_losses, test_losses)

def train_one_epoch(model, train_loader, criterion, optimizer, epoch, device):
    """Train the model for one epoch."""
    model.train()
    train_loss = 0.0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch + 1}/{NUM_EPOCHS}')

    for i, (images, labels) in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        progress_bar.set_postfix({'Train Loss': loss.item()})

    train_loss /= len(train_loader.dataset)
    return train_loss

def evaluate(model, test_loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            test_loss += loss.item()*images.size(0)
    test_loss /= len(test_loader.dataset)

    return test_loss

def save_model(model, model_path):
    """Save the trained model to a file."""
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def plot_losses(train_losses, test_losses):
    """Plot training and testing losses over epochs."""
    epochs = range(1, len(train_losses) + 1)


    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, test_losses, color='red', label='Testing Loss')

    plt.title('Training and Testing Losses Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    initialize_database_if_needed()

    routes, routes_l1 = load_and_preprocess_data()
    train_loader, test_loader = create_datasets_and_loaders(routes, routes_l1)
    train_and_evaluate_model(train_loader, test_loader)