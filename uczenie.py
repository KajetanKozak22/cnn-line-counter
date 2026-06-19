import torch, torch.nn as nn, pandas as pd, cv2, numpy as np, sys, os
from torch.utils.data import DataLoader, TensorDataset

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


class Siec(nn.Module):
    def __init__(self):
        super().__init__()
        self.warstwy = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 40 * 30, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.warstwy(x)

def start():

    df = pd.read_csv(resource_path('labels.csv'))
    X, Y = [], []

    print("Wczytywanie i filtrowanie obrazów (FR-004)...")
    for i in range(len(df)):
        img = cv2.imread(resource_path(f"dataset/img_{i:04d}.png"), 0)
        if img is not None:

            img = cv2.medianBlur(img, 3).astype("float32") / 255.0
            X.append(img)
            Y.append(df.iloc[i, 1])


    X = torch.tensor(np.array(X)).unsqueeze(1)
    Y = torch.tensor(Y, dtype=torch.float32).view(-1, 1)
    
    loader = DataLoader(TensorDataset(X, Y), batch_size=16, shuffle=True)
    
    model = Siec()
    optymalizator = torch.optim.Adam(model.parameters(), lr=0.0005)
    blad_fn = nn.MSELoss() 

    print("\n--- Rozpoczynam trening ---")
    for epoka in range(25):
        suma_bledu = 0
        for batch_x, batch_y in loader:
            predykcja = model(batch_x)
            strata = blad_fn(predykcja, batch_y)
            
            optymalizator.zero_grad()
            strata.backward()
            optymalizator.step()
            
            suma_bledu += strata.item()
        
 
        sredni_blad = suma_bledu / len(loader)
        print(f"Epoka {epoka+1:2d}/25 | Błąd predykcji (MSE): {sredni_blad:.4f}")


    torch.save(model.state_dict(), "model.pth")
    print("\nModel zapisany.")

if __name__ == "__main__":
    start()