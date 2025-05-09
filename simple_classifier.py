import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt # Potrebbe servire per visualizzare i dati

transform = transforms.Compose([
    transforms.ToTensor(), # Converte l'immagine PIL in un Tensor di PyTorch (scala i pixel tra 0 e 1)
    transforms.Normalize((0.5,), (0.5,)) # Normalizza l'immagine (sposta i pixel tra -1 e 1). Il (0.5,) è per la media e deviazione standard per immagini a canale singolo (grayscale).
])

# Scarica e carica il training set
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                            download=True, transform=transform)
# Crea il DataLoader per il training set (per caricare i dati in batch)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, # Scegli la dimensione del batch
                                          shuffle=True) # Mescola i dati per l'addestramento

# Scarica e carica il test set
testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                           download=True, transform=transform)
# Crea il DataLoader per il test set
testloader = torch.utils.data.DataLoader(testset, batch_size=64, # Puoi usare la stessa dimensione batch o diversa per il test
                                         shuffle=False) # Non mescolare i dati di test


# Definisci la Rete Neurale
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Il primo layer lineare prende l'input appiattito (28*28 pixel) e lo mappa a un numero di neuroni nascosti (es. 128)
        self.fc1 = nn.Linear(28 * 28, 128)
        # Funzione di attivazione (ReLU è una scelta comune)
        self.relu = nn.ReLU()
        # Il secondo layer lineare mappa i neuroni nascosti alle 10 classi di output
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Appiattisci l'immagine da 28x28 a un vettore di 784 elementi
        # Il -1 indica che la dimensione del batch viene mantenuta automaticamente
        x = x.view(-1, 28 * 28)
        # Passa l'input appiattito attraverso il primo layer lineare
        x = self.fc1(x)
        # Applica la funzione di attivazione
        x = self.relu(x)
        # Passa il risultato attraverso il secondo layer lineare per ottenere gli output per le classi
        x = self.fc2(x)
        # Nota: Non applichiamo una Softmax qui. La CrossEntropyLoss di PyTorch
        # gestisce la Softmax internamente, quindi l'output sono i "logits"
        return x

# --- Sotto la definizione della classe, crea un'istanza del modello ---

# Imposta il device (CPU o GPU se disponibile)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando device: {device}")

# Crea un'istanza del tuo modello e spostala sul device selezionato
model = SimpleNN().to(device)

print("Modello definito:")
print(model)


# Definisci la Funzione di Perdita
# CrossEntropyLoss è adatta per la classificazione multi-classe
criterion = nn.CrossEntropyLoss()

# Definisci l'Ottimizzatore
# Passa i parametri del modello che devono essere ottimizzati (model.parameters())
# e scegli un learning rate
optimizer = optim.Adam(model.parameters(), lr=0.001) # Puoi provare anche optim.SGD(model.parameters(), lr=0.01)


# --- Ciclo di Addestramento ---

num_epochs = 5 # Decidi quante epoche vuoi addestrare. 5-10 sono un buon inizio.

print(f"Inizio addestramento per {num_epochs} epoche...")

# Loop per le epoche
for epoch in range(num_epochs):
    # Imposta il modello in modalità addestramento (utile per layer come Dropout, Batchnorm, che qui non abbiamo, ma è buona pratica)
    model.train()

    running_loss = 0.0 # Per tenere traccia della loss in questa epoca

    # Loop sui batch di dati del training set
    for i, data in enumerate(trainloader):
        # Ottieni gli input e le etichette dal batch corrente
        inputs, labels = data
        # Sposta i dati sul device (CPU o GPU)
        inputs, labels = inputs.to(device), labels.to(device)

        # === Passi di addestramento per un singolo batch ===

        # 1. Azzera i gradienti dell'ottimizzatore
        optimizer.zero_grad()

        # 2. Forward pass: Calcola l'output del modello
        outputs = model(inputs)

        # 3. Calcola la Loss
        loss = criterion(outputs, labels)

        # 4. Backward pass: Calcola i gradienti
        # Questa chiamata calcola dLoss/d(weights) e li memorizza nei .grad dei parametri
        loss.backward()

        # 5. Aggiorna i pesi del modello
        # L'ottimizzatore usa i gradienti calcolati per fare un passo di discesa
        optimizer.step()

        # === Monitoraggio (opzionale) ===
        running_loss += loss.item() # Aggiungi la loss del batch corrente

        # Stampa la loss ogni tot batch (opzionale, ma utile per vedere l'andamento)
        if i % 100 == 99: # Stampa ogni 100 batch
            print(f'Epoca [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(trainloader)}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0 # Resetta la loss per il prossimo blocco di batch

    print(f"Epoca {epoch + 1} completata.") # Stampa un messaggio a fine epoca

print("Addestramento terminato!")


# --- Fase di Valutazione ---

print("Valutazione del modello sul test set...")

# Imposta il modello in modalità valutazione
# Questo disabilita comportamenti specifici dell'addestramento (es. Dropout, Batchnorm)
model.eval()

# Disabilita il calcolo dei gradienti
# Non ci servono gradienti durante la valutazione/inferenza
with torch.no_grad():
    correct = 0
    total = 0
    # Non c'è bisogno di shufflare il test set, l'ordine non conta per la valutazione
    for data in testloader:
        images, labels = data
        # Sposta i dati sul device (CPU o GPU)
        images, labels = images.to(device), labels.to(device)

        # Forward pass per ottenere le previsioni
        outputs = model(images)

        # Ottieni la classe predetta (l'indice con il valore massimo)
        # torch.max(outputs.data, 1) restituisce (valori, indici). Ci interessa solo gli indici.
        _, predicted = torch.max(outputs.data, 1)

        # Aggiorna il conteggio totale delle immagini
        total += labels.size(0)

        # Aggiorna il conteggio delle predizioni corrette
        correct += (predicted == labels).sum().item() # Confronta predizioni con etichette vere e somma i True (che sono 1)

# Calcola l'accuratezza finale
accuracy = 100 * correct / total

print(f'Accuratezza del modello sul test set: {accuracy:.2f} %')

print("Script completato!")

# --- Salvare il Modello ---

# Definisci il percorso dove salvare il modello
model_save_path = './fashion_mnist_mlp.pth' # Puoi dare il nome che vuoi

print(f"\nSalvataggio del modello in {model_save_path}...")

# Salva lo stato del modello
torch.save(model.state_dict(), model_save_path)

print("Modello salvato!")