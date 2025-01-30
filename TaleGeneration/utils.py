from transformers import AutoTokenizer



# tokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize(data):
    tokenized_data = tokenizer(data["text"], padding="max_length", truncation=True)
    return tokenized_data

def batch_split(data):
    block_size = 128
    concatenated_examples = {k: sum(data[k], []) for k in data.keys()}
    total_length = len(concatenated_examples[list(data.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

#def fine_tunning(model, dataset, trainer, training_args):
 #   training_args = training_args(
  #      "test_trainer",
   #     evaluation_strategy="no",
    #    learning_rate=2e-5,
     #   weight_decay=0.01,
      #  num_train_epochs=200,
       # use_mps_device=True
   # )
 #   trainer = trainer(
  #      model=model,
  #      args=training_args,
  #      train_dataset=dataset,
  #  )
  #  trainer.train()
  #  trainer.save_model("my_model")






