import fasttext
import fasttext.util
import os

# train model if not already trained
if not os.path.exists("model/model.bin"):
    model = fasttext.train_supervised(input="data/dataset.txt")
    model.save_model("model/model.bin")

# load model
model = fasttext.load_model("model/model.bin")

while True:
    # Prompt the user for input and use the model to predict the language
    user_input = input(
        "Enter some text to predict the language (enter 'q' to quit): ")

    # Check if the user wants to quit the program
    if user_input == 'q':
        break

    # Use the model to predict the language of the input
    predicted_language = model.predict(user_input, threshold=0.6)[0][0]

    # Remove the "__label__" prefix from the predicted language
    predicted_language = predicted_language.replace("__label__", "")

    print(f"The predicted language is: {predicted_language}")
