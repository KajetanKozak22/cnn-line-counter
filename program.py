import torch, cv2, pandas as pd, os, sys
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk  

from uczenie import Siec 

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class AplikacjaGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Predykcja Sieci Neuronowej")
        self.root.geometry("750x350")
        self.root.resizable(False, False)
        

        self.model = Siec()
        if not os.path.exists("model.pth"):
            messagebox.showerror("Błąd", "Brak pliku model.pth! Najpierw uruchom test.py.")
            self.root.destroy()
            return
        
        self.model.load_state_dict(torch.load("model.pth", weights_only=True))
        self.model.eval()
        

        self.df = pd.read_csv(resource_path('labels.csv')) if os.path.exists(resource_path('labels.csv')) else None


        self.lewy_panel = tk.Frame(root)
        self.lewy_panel.pack(side="left", fill="both", expand=True, padx=20)

        self.prawy_panel = tk.Frame(root)
        self.prawy_panel.pack(side="right", padx=20, pady=20)

        self.btn_wybierz = tk.Button(self.lewy_panel, text="Wybierz obrazek (.png)", command=self.procesuj_obraz, font=("Arial", 11, "bold"), bg="#4CAF50", fg="white", padx=10, pady=5)
        self.btn_wybierz.pack(pady=20)
        
        self.lbl_plik = tk.Label(self.lewy_panel, text="PLIK: -", font=("Courier", 10))
        self.lbl_plik.pack(anchor="w", pady=2)
        
        self.lbl_prawda = tk.Label(self.lewy_panel, text="PRAWDA: -", font=("Courier", 10))
        self.lbl_prawda.pack(anchor="w", pady=2)
        
        self.lbl_predykcja = tk.Label(self.lewy_panel, text="PREDYKCJA: -", font=("Courier", 10))
        self.lbl_predykcja.pack(anchor="w", pady=2)
        
        self.lbl_wynik = tk.Label(self.lewy_panel, text="WYNIK: -", font=("Courier", 11, "bold"), fg="#107C41")
        self.lbl_wynik.pack(anchor="w", pady=10)

        self.lbl_podglad = tk.Label(self.prawy_panel, text="[ Podgląd obrazu ]", width=320, height=240, bg="#E0E0E0", relief="solid", bd=1)
        self.lbl_podglad.pack()

    def procesuj_obraz(self):
        sciezka_pliku = filedialog.askopenfilename(
            initialdir=resource_path("dataset") if os.path.exists(resource_path("dataset")) else ".",
            title="Wybierz plik obrazu",
            filetypes=(("Pliki PNG", "*.png"), ("Wszystkie pliki", "*.*"))
        )
        
        if not sciezka_pliku:
            return

        nazwa = os.path.basename(sciezka_pliku)
        img = cv2.imread(sciezka_pliku, 0)
        
        if img is None:
            messagebox.showerror("Błąd", f"Nie udało się wczytać pliku: {nazwa}")
            return

        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img = pil_img.resize((320, 240), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil_img)
        self.lbl_podglad.config(image=tk_img, text="")
        self.lbl_podglad.image = tk_img 

        img_p = cv2.medianBlur(img, 3).astype("float32") / 255.0
        tensor = torch.tensor(img_p).reshape(1, 1, 240, 320)

        with torch.no_grad():
            predykcja_surowa = self.model(tensor).item()
            wynik_koncowy = round(max(0, predykcja_surowa))

        prawda = "Nieznana"
        if self.df is not None:
            try:
                idx = int(nazwa.split('_')[1].split('.')[0])
                prawda = self.df.iloc[idx, 1]
            except: pass

        self.lbl_plik.config(text=f"PLIK:      {nazwa}")
        self.lbl_prawda.config(text=f"PRAWDA:    {prawda} (Liczba linii z CSV)")
        self.lbl_predykcja.config(text=f"PREDYKCJA: {predykcja_surowa:.4f} (Dokładność sieci)")
        self.lbl_wynik.config(text=f"WYNIK:     {wynik_koncowy} (Zliczone odcinki)")

if __name__ == "__main__":
    root = tk.Tk()
    app = AplikacjaGUI(root)
    root.mainloop()