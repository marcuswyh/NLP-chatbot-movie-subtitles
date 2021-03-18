import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import pickle
from multi_rake import Rake
import re
import os

size = 10000

sentences = []
questions, ans = [], []
files = ["movie_subtitles_en.txt"]
rake = Rake()

for f in files:
    # get all sentences
    with open(f, 'r', errors='ignore') as file:
        for line in file.readlines():
            if len(sentences) == 0:
                sentences.append(line.lower())
            else:
                if line.lower() != sentences[-1]:
                    sentences.append(line.lower())
    
    # separate sentences to questions and responses
    for idx, s in enumerate(sentences[:-1]):
        questions.append(sentences[idx])
        ans.append(sentences[idx + 1])
    
    sentences = []

# further refine question and answer sentences
del questions[1::2]
del ans[1::2]

# trim data size to defined size
questions =  questions[:size]
ans = ans[:size]

# create keywords for each sentence using Rake
key = []
for idx, q in enumerate(questions):
    keyword = rake.apply(" ".join(re.findall(r"\w+", q.lower())))
    if len(keyword) != 0:
        key.append((keyword[0])[0])
    else:
        key.append((line.split())[0])
            


try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    # tokenize all question sentences and attach keyword labels based on index
    for idx, q in enumerate(questions):
        wrds = nltk.word_tokenize(q)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(key[idx])

        if key[idx] not in labels:
            labels.append(key[idx])

    # stemming and creating word vocab
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    # save processed dataset information
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# declare tensorflow training parameters
tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# if model already exists, use it
# otherwise, train using processed data
if os.path.exists("model.tflearn.meta"):
    model.load("model.tflearn")
else:
    model.fit(training, output, n_epoch=150, batch_size=8, show_metric=True)
    model.save("model.tflearn")

# bag of words approach for model prediction
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

# personality keywords and response traits
personality = { "name": "My name is TensorBro.",
                "self": "I am a chatbot.",
                "place": "I live in your computer :)",
                "hate": "I hate being a chatbot.",
                "love": "I love ice-cream, but i can't eat so i want to be a human in the future.",
                "movie": "My favorite movie is Frozen!"}

personality_key = { "name": ["what", "your", "name"],
                "self": ["who", "you"],
                "place": ["where", "you", "live", "from"],
                "hate": ["hate", "you"],
                "love": [ "you", "like", "love"],
                "movie": ["favorite", "your", "movie"]}

def chat():
    print("Start talking with the bot (press enter to stop)!")
    isquestion = False
    savedInput = None
    output_file = open("output_convo.txt", 'a+')
    while True:
        inp = input("You: ")
        if inp.lower() == "":
            break

        output_file.write('HUMAN ++++ ' + inp + '\n')

        # gets answer from user after bot asked a question
        if isquestion:
            savedInput = inp
            isquestion = False
        
        # if user asks bot a question
        if (inp.strip())[-1] == "?":
            # if there is an answer from user referring to existing bot question
            if savedInput != None:
                print ("BOT: You said '"+savedInput+"'\n")
                output_file.write('BOT ++++ You said ' + savedInput + '\n')
                savedInput = None
                continue
            
            # detects if user is asking question in relation to bot's personality
            p_flag = False
            for key in personality_key:
                count = 0
                # detects if user input containes personality keywords
                for word in re.findall(r"\w+",inp.lower()):
                    if word in personality_key[key]:
                        count+=1/len(personality_key[key])
                if count >= 0.6:
                    print ("BOT: "+personality[key]+"\n")
                    output_file.write('BOT ++++ ' + personality[key] + '\n')
                    p_flag = True
                    break
            
            if p_flag:
                continue

        # bag of words response prediction
        results = model.predict([bag_of_words(inp, words)])
        # get 3 best answers
        results_index = numpy.argpartition(results[0], -3)[-3:]
        # choose one best answer out of 3
        tag = labels[random.choice(results_index)]

        # get index of answer based on keyword tag
        npLabels = numpy.array(labels)
        index = numpy.where(npLabels == tag)[0]

        # print response using obtained index
        print("BOT: ",ans[index[0]])
        output_file.write('BOT ++++ ' + ans[index[0]] + '\n')

        # to detect if bot is asking user a question
        if (ans[index[0]].strip())[-1] == "?":
            isquestion = True

    output_file.write('=============================================\n')
    output_file.close()

chat()