# zaimportowanie biblioteki fasttext
import fasttext

# trening modelu
model = fasttext.train_supervised(input="data/dataset.txt")

# zapis modelu
model.save_model("model/model.bin")

# odczyt modelu
model = fasttext.load_model("model/model.bin")

# predykcja języka
predicted_language = model.predict(user_input)[0][0]
