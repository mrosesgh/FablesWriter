from datasets import Dataset
import requests
from utils import tokenize
from utils import batch_split

f = open("pg2591.txt", "w")
f.write(requests.get("https://www.gutenberg.org/cache/epub/50316/pg50316.txt").text)
f.close()

# Extract Fable's titles from the text

fablesTitles = []
titles = False

nonFables = ["As Essay on the Life and Works of Jean de la Fontaine", "The Life of Æsop, the Phrygian", "Dedication to Monseigneur the Dauphin", "Preface", "To Monseigneur the Dauphin", ""]

f = open("pg2591.txt", "r")
for line in f.readlines():
    text = line.strip()
    if "LIST OF FULL-PAGE ILLUSTRATIONS." == text:
        break
    if titles:
        if text not in nonFables:
            a_title = text.upper()
            a_title = str(a_title+'.')
            fablesTitles.append(a_title)
    if "CONTENTS" == text:
        titles = True
f.close()

# Create a dictionary with titles and fables

stories = {title: [] for title in fablesTitles}
f = open("pg2591.txt", "r")
title = None

for line in f.readlines():
    text = line.strip()
    if "End of Project Gutenberg's The Fables of La Fontaine, by Jean de la Fontaine" == text:
        break
    if text in stories:
        title = text
        continue
    if title:
        stories[title].append(text)
f.close()

# Create dataset

dataset = {
    "text": []
}
for story in stories:
    dataset["text"].append(" ".join(list(filter(lambda x: x!="", stories[story]))))
dataset["text"] = list(filter(lambda x: x!="", dataset["text"]))
##################
fablesDataset = Dataset.from_dict(dataset)


# preprocess the dataset

token_fablesDataset = fablesDataset.map(tokenize, batched=True, remove_columns=["text"])
#############################
proc_fablesDataset = token_fablesDataset.map(
    batch_split,
    batched=True,
    batch_size=1000,
    num_proc=4,
    )
