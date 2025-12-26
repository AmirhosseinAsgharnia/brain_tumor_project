import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score



def main():
    training_path = "./training_data"
    testing_path = "./testing_data"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    class BayesianConv2d(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, prior_sigma=1.0):
            super().__init__()
            if isinstance(kernel_size, int):
                kh, kw = kernel_size, kernel_size
            else:
                kh, kw = kernel_size
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = (kh, kw)
            self.stride = stride
            self.padding = padding
            self.dilation = dilation
            self.groups = groups
            self.has_bias = bias
            self.prior_sigma = float(prior_sigma)

            weight_shape = (out_channels, in_channels // groups, kh, kw)
            self.weight_mu = nn.Parameter(torch.empty(weight_shape))
            self.weight_rho = nn.Parameter(torch.empty(weight_shape))

            if bias:
                self.bias_mu = nn.Parameter(torch.empty(out_channels))
                self.bias_rho = nn.Parameter(torch.empty(out_channels))
            else:
                self.register_parameter("bias_mu", None)
                self.register_parameter("bias_rho", None)

            self.reset_parameters()

        def reset_parameters(self):
            nn.init.kaiming_uniform_(self.weight_mu, a=np.sqrt(5))
            nn.init.constant_(self.weight_rho, -5.0)
            if self.has_bias:
                fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1] / self.groups
                bound = 1.0 / np.sqrt(fan_in)
                nn.init.uniform_(self.bias_mu, -bound, bound)
                nn.init.constant_(self.bias_rho, -5.0)

        def _sigma(self, rho):
            return F.softplus(rho)

        def kl(self):
            prior_var = self.prior_sigma ** 2
            w_sigma = self._sigma(self.weight_rho)
            w_var = w_sigma ** 2
            kl_w = 0.5 * torch.sum((w_var + self.weight_mu ** 2) / prior_var - 1.0 + torch.log(prior_var / (w_var + 1e-12)))

            if not self.has_bias:
                return kl_w

            b_sigma = self._sigma(self.bias_rho)
            b_var = b_sigma ** 2
            kl_b = 0.5 * torch.sum((b_var + self.bias_mu ** 2) / prior_var - 1.0 + torch.log(prior_var / (b_var + 1e-12)))
            return kl_w + kl_b

        def forward(self, x):
            w_sigma = self._sigma(self.weight_rho)
            w = self.weight_mu + w_sigma * torch.randn_like(self.weight_mu)

            if self.has_bias:
                b_sigma = self._sigma(self.bias_rho)
                b = self.bias_mu + b_sigma * torch.randn_like(self.bias_mu)
            else:
                b = None

            return F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups)

    class BayesianLinear(nn.Module):
        def __init__(self, in_features, out_features, bias=True, prior_sigma=1.0):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.has_bias = bias
            self.prior_sigma = float(prior_sigma)

            self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
            self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))

            if bias:
                self.bias_mu = nn.Parameter(torch.empty(out_features))
                self.bias_rho = nn.Parameter(torch.empty(out_features))
            else:
                self.register_parameter("bias_mu", None)
                self.register_parameter("bias_rho", None)

            self.reset_parameters()

        def reset_parameters(self):
            nn.init.kaiming_uniform_(self.weight_mu, a=np.sqrt(5))
            nn.init.constant_(self.weight_rho, -5.0)
            if self.has_bias:
                bound = 1.0 / np.sqrt(self.in_features)
                nn.init.uniform_(self.bias_mu, -bound, bound)
                nn.init.constant_(self.bias_rho, -5.0)

        def _sigma(self, rho):
            return F.softplus(rho)

        def kl(self):
            prior_var = self.prior_sigma ** 2
            w_sigma = self._sigma(self.weight_rho)
            w_var = w_sigma ** 2
            kl_w = 0.5 * torch.sum((w_var + self.weight_mu ** 2) / prior_var - 1.0 + torch.log(prior_var / (w_var + 1e-12)))

            if not self.has_bias:
                return kl_w

            b_sigma = self._sigma(self.bias_rho)
            b_var = b_sigma ** 2
            kl_b = 0.5 * torch.sum((b_var + self.bias_mu ** 2) / prior_var - 1.0 + torch.log(prior_var / (b_var + 1e-12)))
            return kl_w + kl_b

        def forward(self, x):
            w_sigma = self._sigma(self.weight_rho)
            w = self.weight_mu + w_sigma * torch.randn_like(self.weight_mu)

            if self.has_bias:
                b_sigma = self._sigma(self.bias_rho)
                b = self.bias_mu + b_sigma * torch.randn_like(self.bias_mu)
            else:
                b = None

            return F.linear(x, w, b)

    class BayesianCNN_Class(nn.Module):
        def __init__(self, num_of_classes=4, prior_sigma=1.0):
            super().__init__()

            self.feature = nn.Sequential(
                BayesianConv2d(3, 3, kernel_size=3, stride=2, padding=1, groups=3, prior_sigma=prior_sigma),
                BayesianConv2d(3, 32, kernel_size=1, stride=1, padding=0, prior_sigma=prior_sigma),
                nn.ReLU(inplace=True),

                BayesianConv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32, prior_sigma=prior_sigma),
                BayesianConv2d(32, 32, kernel_size=1, stride=1, padding=0, prior_sigma=prior_sigma),
                nn.ReLU(inplace=True),

                BayesianConv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32, prior_sigma=prior_sigma),
                BayesianConv2d(32, 64, kernel_size=1, stride=1, padding=0, prior_sigma=prior_sigma),
                nn.ReLU(inplace=True),

                BayesianConv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=64, prior_sigma=prior_sigma),
                BayesianConv2d(64, 128, kernel_size=1, stride=1, padding=0, prior_sigma=prior_sigma),
                nn.ReLU(inplace=True),

                BayesianConv2d(128, 128, kernel_size=3, stride=2, padding=1, groups=128, prior_sigma=prior_sigma),
                BayesianConv2d(128, 256, kernel_size=1, stride=1, padding=0, prior_sigma=prior_sigma),
                nn.ReLU(inplace=True),

                BayesianConv2d(256, 256, kernel_size=3, stride=2, padding=1, groups=256, prior_sigma=prior_sigma),
                BayesianConv2d(256, 512, kernel_size=1, stride=1, padding=0, prior_sigma=prior_sigma),
                nn.ReLU(inplace=True),

                BayesianConv2d(512, 512, kernel_size=3, stride=2, padding=1, groups=512, prior_sigma=prior_sigma),
                BayesianConv2d(512, 1024, kernel_size=1, stride=1, padding=0, prior_sigma=prior_sigma),
                nn.ReLU(inplace=True),
            )

            self.avepooling = nn.AdaptiveAvgPool2d((1, 1))

            self.classifier = nn.Sequential(
                BayesianLinear(1024, 256, prior_sigma=prior_sigma),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                BayesianLinear(256, num_of_classes, prior_sigma=prior_sigma),
            )

        def forward(self, x):
            x = self.feature(x)
            x = self.avepooling(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

        def kl_loss(self):
            total = 0.0
            for m in self.modules():
                if hasattr(m, "kl") and callable(getattr(m, "kl")):
                    total = total + m.kl()
            return total

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    train_ds_full = datasets.ImageFolder(training_path, transform=train_tf)
    test_ds = datasets.ImageFolder(testing_path, transform=test_tf)

    class_names = train_ds_full.classes
    num_classes = len(class_names)

    val_ratio = 0.1
    val_size = int(len(train_ds_full) * val_ratio)
    train_size = len(train_ds_full) - val_size
    train_ds, val_ds = random_split(train_ds_full, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=1, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=1, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=1, pin_memory=False)

    model = BayesianCNN_Class(num_of_classes=num_classes, prior_sigma=1.0).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    def elbo_loss(model, logits, y, kl_scale):
        nll = F.cross_entropy(logits, y, reduction="mean")
        kl = model.kl_loss()
        return nll + kl_scale * kl, nll.detach(), kl.detach()

    def mc_predict_batch(model, x, T=10):
        model.train()
        probs = []
        with torch.no_grad():
            for _ in range(T):
                logits = model(x)
                probs.append(F.softmax(logits, dim=1).unsqueeze(0))
        probs = torch.cat(probs, dim=0)
        mean_prob = probs.mean(dim=0)
        eps = 1e-12
        entropy = -(mean_prob * torch.log2(mean_prob.clamp_min(eps))).sum(dim=1)
        return mean_prob, entropy

    def eval_loader_bayes(model, loader, T=10):
        ys = []
        ps = []
        model.train()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                mean_prob, _ = mc_predict_batch(model, x, T=T)
                pred = torch.argmax(mean_prob, dim=1)
                ys.append(y.detach().cpu().numpy())
                ps.append(pred.detach().cpu().numpy())
        y_true = np.concatenate(ys)
        y_pred = np.concatenate(ps)
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        return acc, prec, rec, f1, cm, y_true, y_pred

    best_val_acc = -1.0
    best_state = None

    epochs = 100
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0
        correct = 0
        total_loss = 0.0
        total_nll = 0.0
        total_kl = 0.0

        kl_scale = 1.0 / max(1, len(train_loader))

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss, nll, kl = elbo_loss(model, logits, y, kl_scale)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_nll += float(nll.item()) * x.size(0)
            total_kl += float(kl.item()) * x.size(0)

            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)

        train_acc = correct / max(1, total)
        val_acc, val_prec, val_rec, val_f1, _, _, _ = eval_loader_bayes(model, val_loader, T=10)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v.to(device) if torch.is_tensor(v) else v for k, v in best_state.items()})

    val_acc, val_prec, val_rec, val_f1, val_cm, val_y, val_p = eval_loader_bayes(model, val_loader, T=10)
    test_acc, test_prec, test_rec, test_f1, test_cm, test_y, test_p = eval_loader_bayes(model, test_loader, T=10)

    print("Classes:", class_names)
    print("Best Val Acc:", best_val_acc)
    print("VAL  Acc/Prec/Rec/F1:", val_acc, val_prec, val_rec, val_f1)
    print("TEST Acc/Prec/Rec/F1:", test_acc, test_prec, test_rec, test_f1)
    print("VAL Confusion Matrix:\n", val_cm)
    print("TEST Confusion Matrix:\n", test_cm)
    print("TEST Classification Report:\n", classification_report(test_y, test_p, target_names=class_names, zero_division=0))

    def mc_single_image_report(model, dataset, idx=None, T=50):
        if idx is None:
            idx = random.randint(0, len(dataset) - 1)
        img, y = dataset[idx]
        x = img.unsqueeze(0).to(device)
        mean_prob, entropy = mc_predict_batch(model, x, T=T)
        pred = int(torch.argmax(mean_prob, dim=1).item())
        return idx, int(y), pred, float(entropy.item()), mean_prob.squeeze(0).detach().cpu().numpy()

    idx, y_true, y_pred, ent, probs = mc_single_image_report(model, test_ds, idx=None, T=50)
    print("Sample idx:", idx)
    print("True label:", class_names[y_true])
    print("Pred label:", class_names[y_pred])
    print("Entropy:", ent)
    print("Mean prob:", probs)

    ckpt = {
        "model_state": model.state_dict(),
        "classes": class_names,
        "best_val_acc": best_val_acc,
    }
    torch.save(ckpt, "bayesian_dwcnn_full_vi.pt")
    print("Saved: bayesian_dwcnn_full_vi.pt")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()