import random
import json
import pickle
import time
import numpy as np
import requests
import PySimpleGUI as sg
from tkinter import *
import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import load_model

lemmatizer = WordNetLemmatizer()
inntents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = load_model('model.h5')
localtime = time.asctime(time.localtime(time.time()))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent':classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


root = Tk()
root.title("Albot")
BG_GRAY = "#ABB2B9"
BG_COLOR = "#34282C"
TEXT_COLOR = "#EAECEE"

FONT = "Arial 14"
FONT_BOLD = "Arial 14 bold"
lable1 = Label(root, bg=BG_COLOR, fg=TEXT_COLOR, text="Albot",font=FONT_BOLD, pady=10,width=30,height=1).grid(row=0)
txt = Text(root,bg=BG_COLOR, fg=TEXT_COLOR,font=FONT, width=100)
txt.grid(row=1,column=0,columnspan=2)

scrollbar = Scrollbar(txt)
scrollbar.place(relheight=1,relx=0.974)

e = Entry(root,bg="#2C3E50",fg=TEXT_COLOR, font=FONT,width=55)
e.grid(row=2, column=0)




def send_message():
    message = e.get()
    e.delete(0, END)
    txt.insert(END, "You -> " + message + "\n")
    ints = predict_class(message)
    resp = get_response(ints, inntents)
    txt.insert(END, "Al -> " + resp + "\n")




send_button = Button(root, text="Send", font=FONT_BOLD, bg=BG_GRAY, command=send_message)
send_button.grid(row=2, column=1)

root.mainloop()






#print(localtime)
#print("start talking")

