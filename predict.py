import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image # Per caricare immagini esterne, se vuoi


# --- 1. Definire la stessa struttura del Modello ---
# QUESTA CLASSE DEVE ESSERE IDENTICA A QUELLA USATA IN simple_classifier.py
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128) # Assicurati che le dimensioni siano le stesse
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10) # Assicurati che le dimensioni siano le stesse (10 classi)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Definisci le classi di FashionMNIST (l'ordine è importante!)
classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Imposta il device (CPU o GPU) - Deve corrispondere a dove caricherai il modello
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando device: {device}")

# --- 2. Creare un'istanza del Modello e Caricare lo Stato Addestrato ---

# Crea un'istanza del modello
model = SimpleNN()

# Definisci il percorso del modello salvato
model_save_path = './fashion_mnist_mlp.pth' # Deve corrispondere al nome usato per salvare

# Carica lo stato del modello (pesi e bias)
model.load_state_dict(torch.load(model_save_path, map_location=device)) # map_location gestisce se il modello era su GPU/CPU

# Sposta il modello sul device selezionato
model.to(device)

# Imposta il modello in modalità valutazione
model.eval()

print(f"Modello '{model_save_path}' caricato con successo.")

# --- 3. Preparare l'Immagine Singola per la Previsione ---

# Puoi caricare un'immagine esterna o prenderne una dal test set come esempio.
# Useremo un esempio dal test set per semplicità, ma puoi adattare per caricare da file.

# Carica il test set solo per prendere un esempio (assicurati che le trasformazioni siano le stesse!)
testset_predict = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                    download=True, transform=transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.5,), (0.5,))
                                                    ]))


idx = 100 # Scegli un indice a caso (o uno specifico) dal test set
sample_image, sample_label = testset_predict[idx] # Prendi l'immagine e l'etichetta vera (per confronto)

# Mostra l'immagine (opzionale)
plt.imshow(sample_image.squeeze().numpy(), cmap='gray')
plt.title(f"Immagine per la previsione (Etichetta vera: {classes[sample_label]})")
plt.show()


# Prepara l'immagine: aggiungi dimensione batch e spostala sul device
sample_image = sample_image.unsqueeze(0).to(device)


# --- 4. Eseguire la Previsione ---

print("\nEsecuzione della previsione...")

with torch.no_grad(): # Disabilita il calcolo dei gradienti
    # Passa l'immagine al modello
    output = model(sample_image)

    # Ottieni le probabilità e la classe predetta
    probabilities = torch.softmax(output, dim=1)
    confidence, predicted_index = torch.max(probabilities, 1)

# Ottieni il nome della classe predetta e la sua confidenza
predicted_class = classes[predicted_index.item()]
confidence_percentage = confidence.item() * 100

print(f"Etichetta vera: {classes[sample_label]}") # Mostra l'etichetta vera per confronto
print(f"Predizione del modello: {predicted_class}")
print(f"Confidenza: {confidence_percentage:.2f} %")

print("\nScript di previsione completato!")