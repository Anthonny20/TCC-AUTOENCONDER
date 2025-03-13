import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM, PeakSignalNoiseRatio as PSNR

def compute_reconstruction_metrics(model, dataloader, device):
    model.eval()
    mse_loss = torch.nn.MSELoss()
    ssim = SSIM().to(device)
    psnr = PSNR().to(device)
    mse, ssim_val, psnr_val = 0.0, 0.0, 0.0

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            if isinstance(model, (VAE_Linear, VAE_Conv)):
                reconstructed, _, _ = model(x)
            else:
                reconstructed, _ = model(x)

            mse += mse_loss(reconstructed, x).item()
            ssim_val += ssim(reconstructed, x).item()
            psnr_val += psnr(reconstructed, x).item()
    
    return {
        'MSE': mse / len(dataloader),
        'SSIM': ssim_val / len(dataloader),
        'PSNR': psnr_val / len(dataloader)

    }

def evaluate_classification(model, train_loader, test_loader, device):
    X_train, y_train = extract_features(model, train_loader, device)
    X_test, y_test = extract_features(model, test_loader, device)

    clf = SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test,y_pred)

def extract_features(model, dataloader, device):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            if isinstance(model, (VAE_Linear, VAE_Conv)):
                z, _, _ =model.encoder(x)
            else:
                z = model.encoder(x)
            features.append(z.cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(features), np.concatenate(labels)