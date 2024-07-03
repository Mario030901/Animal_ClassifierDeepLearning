import os
import cv2 as cv
from sklearn.model_selection import train_test_split
from keras.src.models import Sequential
from keras.src.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.src.optimizers import Adam
from keras.src.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def create_dataset(path, size):
    # Liste per le immagini e le etichette
    images = []
    labels = []

    # Ottiene le etichette dalle sotto-cartelle
    class_names = os.listdir(path)

    # Itera su ogni sotto-cartella (etichetta)
    for label, class_name in enumerate(class_names):
        class_path = os.path.join(path, class_name)
        
        # Controlla se il percorso è una directory
        if os.path.isdir(class_path):
            
            # Itera su ogni immagine nella sotto-cartella
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                
                # Controlla se il file è un'immagine
                if os.path.isfile(image_path):
                    
                    # Ridimensiona l'immagine
                    resized_image = cv.resize(cv.imread(image_path), (size, size))
                    
                    # Aggiunge l'immagine e l'etichetta alle rispettive liste
                    images.append(resized_image/255)
                    labels.append(label)

    return np.array(images), np.array(labels)
# Funzione per verificare se un file è un'immagine valida
def is_valid_image(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()  # Verifica che sia un'immagine valida
        return True
    except (IOError, SyntaxError) as e:
        print(f"File non valido: {filepath} - {e}")
        return False

# Funzione per pulire le cartelle da immagini non valide
def clean_dataset(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            if not is_valid_image(filepath):
                print(f"Rimuovendo file non valido: {filepath}")
                os.remove(filepath)

# Configurazione del percorso del dataset
dataset_dir = r'C:\Users\MarioCampana\OneDrive - ITS Angelo Rizzoli\Desktop\UFS12Nuovo\raw-img'
#dataset_dir = r'C:\Users\MarioCampana\Downloads\archive\raw-img'

# Pulizia delle cartelle del dataset da immagini non valide
clean_dataset(dataset_dir)

size = 128
images, labels = create_dataset(dataset_dir, size)

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.10, shuffle=True, stratify=labels)


# Definizione del modello 
model = Sequential(
    layers=[
        Conv2D(50, (3, 3), activation="relu", input_shape=(size, size, 3)),
        Conv2D(100, (3, 3), activation="relu"),
        MaxPooling2D(),
        Conv2D(75, (3, 3), activation="relu"),
        #max 
        MaxPooling2D(),
        #conv  
        Conv2D(125, (3, 3), activation="relu"),
        Flatten(),
        Dense(200, activation="relu"),
        Dropout(0.50),
        Dense(200, activation="relu"), #-----
        Dropout(0,50),#------
        Dense(10, activation="softmax")
        ],
    name="ModelloAnimali"
)

model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

# Creazione di un callback EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True) # EarlyStopping aggiunto

epoche = 10
history = model.fit(
x_train, y_train, 
    batch_size=64, 
    epochs=epoche, 
    validation_split=0.20,
    callbacks=[early_stopping] # EarlyStopping 
)





#epoche = 20
#history = model.fit(x_train, y_train, batch_size=64, epochs=epoche, validation_split=0.20)#il 20% del mio train diventa validation


model.save("C:\\Users\\MarioCampana\\OneDrive - ITS Angelo Rizzoli\\Desktop\\UFS12Nuovo\\Modello_Allenato2.keras")

# Visualizzazione della storia dell'addestramento
# Visualizzazione della storia dell'addestramento
history_dict = history.history
xx = np.arange(1, len(history_dict['loss']) + 1)  # Aggiorna la lunghezza di xx in base alla storia

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(xx, history_dict['loss'], c='r', label='Training loss')
plt.plot(xx, history_dict['val_loss'], c='blue', label='Validation loss')
plt.xlabel('Epoche')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(xx, history_dict['accuracy'], c='r', label='Training accuracy')
plt.plot(xx, history_dict['val_accuracy'], c='blue', label='Validation accuracy')
plt.xlabel('Epoche')
plt.ylabel('Accuratezza')
plt.legend()


# Calcolo e visualizzazione della matrice di confusione
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
class_labels = sorted(np.unique(labels))  # Ordina le etichette delle classi
cm = confusion_matrix(y_test, y_pred_classes, labels=class_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matrice di Confusione")
plt.show()


#cos'è il batch 
#cosa fa la fit 
#le epoche cosa sono 
#la mia rete neurale la sudivisione neuroni
#come ho diviso validation test train 
#model=load_model("dove è salvato")