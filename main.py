import argparse
import os

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from scipy.io import loadmat


# --- 1. 参数设置 ---
parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs-mnist', type=int, default=10, metavar='N',
                    help='number of epochs to train for MNIST (default: 10)')
parser.add_argument('--epochs-frey', type=int, default=100, metavar='N',
                    help='number of epochs to train for Frey Face (default: 100)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()


# --- 2. 环境与设备设置 ---
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")


# --- 3. 模型定义 ---

# 3.1 VAE for MNIST (Binary Likelihood)
class VAE_Binary(nn.Module):
    def __init__(self, input_dim=784, h_dim=400, z_dim=20):
        super(VAE_Binary, self).__init__()
        self.fc1 = nn.Linear(input_dim, h_dim)
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# 3.2 VAE for Frey Face (Gaussian Likelihood)
class VAE_Gaussian(nn.Module):
    def __init__(self, input_dim=560, h_dim=200, z_dim=30):
        super(VAE_Gaussian, self).__init__()
        self.fc1 = nn.Linear(input_dim, h_dim)
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 560))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# --- 4. 损失函数 ---
def loss_function_bce(recon_x, x, mu, logvar, beta=1.0):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD

def loss_function_mse(recon_x, x, mu, logvar, beta=1.0):
    MSE = F.mse_loss(recon_x, x.view(-1, 560), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + beta * KLD

# --- 5. 主执行函数 ---
def main():
    if not os.path.exists('results_mnist'):
        os.makedirs('results_mnist')
    if not os.path.exists('results_freyface'):
        os.makedirs('results_freyface')

    # --- PART 1: MNIST (Binary Likelihood) ---
    print("\n--- Starting Part 1: VAE on MNIST (Binary Likelihood) ---")
    
    mnist_train_loader = DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=False)

    model_mnist = VAE_Binary().to(device)
    optimizer_mnist = optim.Adam(model_mnist.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs_mnist + 1):
        model_mnist.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(mnist_train_loader):
            data = data.to(device)
            optimizer_mnist.zero_grad()
            recon_batch, mu, logvar = model_mnist(data)
            loss = loss_function_bce(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer_mnist.step()
            if batch_idx % args.log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(mnist_train_loader.dataset)}'
                      f' ({100. * batch_idx / len(mnist_train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
        print(f'====> Epoch: {epoch} Average loss: {train_loss / len(mnist_train_loader.dataset):.4f}')

    # --- PART 2: Frey Face (Gaussian Likelihood) ---
    print("\n--- Starting Part 2: VAE on Frey Face (Gaussian Likelihood) ---")
    
    frey_face_file = "frey_rawface.mat"
    frey_faces = loadmat(frey_face_file)["ff"].T.astype('float32') / 255.0
    frey_faces_tensor = torch.from_numpy(frey_faces)
    frey_dataset = TensorDataset(frey_faces_tensor)
    frey_loader = DataLoader(frey_dataset, batch_size=args.batch_size, shuffle=True)
    
    model_frey = VAE_Gaussian().to(device)
    optimizer_frey = optim.Adam(model_frey.parameters(), lr=args.lr)
    
    for epoch in range(1, args.epochs_frey + 1):
        model_frey.train()
        train_loss = 0
        for batch_idx, (data,) in enumerate(frey_loader):
            data = data.to(device)
            optimizer_frey.zero_grad()
            recon_batch, mu, logvar = model_frey(data)
            loss = loss_function_mse(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer_frey.step()
        
        if epoch % 10 == 0 or epoch == args.epochs_frey:
            print(f'====> Epoch: {epoch} Average loss: {train_loss / len(frey_loader.dataset):.4f}')


  # --- 可视化 ---
    print("\n--- Visualization (Original, Reconstruction, Residual) ---")

    # MNIST 结果可视化
    data_batch, _ = next(iter(mnist_test_loader))
    indices = torch.randperm(data_batch.size(0))[:10]
    samples_X = data_batch[indices].to(device)
    recon_X, _, _ = model_mnist(samples_X)
    recon_X_reshaped = recon_X.view(10, 1, 28, 28)
    residual = torch.abs(samples_X - recon_X_reshaped)
    comparison = torch.cat([samples_X,
                            recon_X_reshaped,
                            residual])
    save_image(comparison.cpu(),
               'results_mnist/reconstruction_comparison.png',
               nrow=10)
    print("Saved MNIST originals, reconstructions, and residuals to 'results_mnist/reconstruction_comparison.png'")

    # Frey Face 结果可视化
    data_batch, = next(iter(frey_loader))
    indices = torch.randperm(data_batch.size(0))[:10]
    samples_X = data_batch[indices].to(device)
    recon_X, _, _ = model_frey(samples_X)
    img_shape = (28, 20)
    recon_X_reshaped = recon_X.view(10, 1, *img_shape)
    residual = torch.abs(samples_X.view(10, 1, *img_shape) - recon_X_reshaped)
    comparison = torch.cat([samples_X.view(10, 1, *img_shape),
                            recon_X_reshaped,
                            residual])
    save_image(comparison.cpu(),
               'results_freyface/reconstruction_comparison.png',
               nrow=10)
    print("Saved Frey Face originals, reconstructions, and residuals to 'results_freyface/reconstruction_comparison.png'")


if __name__ == '__main__':
    main()