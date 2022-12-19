import fasttext
import os
import fasttext.util


# train model if not already trained
if not os.path.exists("model\model3.bin"):
    model = fasttext.train_supervised(input='data\dataset.txt')
    model.save_model("model\model3.bin")

# load model
model = fasttext.load_model("model\model3.bin")


# with open('src\language-codes_csv.csv', mode='r') as infile:
#     reader = csv.reader(infile)
#     land_dict = {rows[0]: rows[1] for rows in reader}

# print(land_dict)


while True:
    # Prompt the user for input and use the model to predict the language
    user_input = input(
        "Enter some text to predict the language (enter 'q' to quit): ")

    # Check if the user wants to quit the program
    if user_input == 'q':
        break

    # Use the model to predict the language of the input
    # predicted_language = model.predict(user_input, threshold=0.6)[0][0]
    predicted_language = model.predict(user_input, k=3)

    predicted_language_accuracy = predicted_language[1]
    predicted_language = predicted_language[0]

    for l, a in zip(predicted_language, list(predicted_language_accuracy)):
        # Remove the "__label__" prefix from the predicted language
        l = l.replace("__label__", "")
        # language = land_dict[l]
        print("The predicted language is: {}\t Probability: {:.2f}%".format(
            l, a*100))
