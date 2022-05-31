from pyth_imports import *


# For paraphrases
data_para_train_loc = "msr-paraphrase-corpus/msr_paraphrase_train.txt"
data_para_test_loc = "msr-paraphrase-corpus/msr_paraphrase_test.txt"
data_para_train = pd.read_csv(data_para_train_loc, sep="\t")
data_para_test = pd.read_csv(data_para_test_loc, sep="\t")


out_analogies_train = "msr-paraphrase-corpus/analogies_paraphrases_train_4M.csv"
# out_analogies_dev = "/moredata/stergos/analogies/analogies_learning_sets/paraphrases/analogies_paraphrases_dev.csv"
out_analogies_test = "msr-paraphrase-corpus/analogies_paraphrases_test_400K.csv"
out_analogies_valid = "msr-paraphrase-corpus/analogies_paraphrases_valid_400K.csv"
# out_sentences = "/moredata/stergos/analogies/analogies_learning_sets/paraphrases/sentences_train.csv"

NB_ANALOGIES_TRAIN = 4000000
NB_ANALOGIES_VALID = 400000
NB_ANALOGIES_TEST = 400000

seen = set()
max_positive = max_negative = NB_ANALOGIES_TRAIN
positive = negative = 0
    
# For paraphrases
with open(out_analogies_train, 'w+') as out :

  while True :
      random = data_para_train.sample(n=2) 
      y = int(int(random.iloc[0]['Quality']) == int(random.iloc[1]['Quality']) and int(random.iloc[1]['Quality']) == 1)
      
      analogy = str(random.iloc[0]['#1 ID']) + '|' + str(random.iloc[0]['#2 ID']) + '|' + str(random.iloc[1]['#1 ID']) + '|' + str(random.iloc[1]['#2 ID']) + '|' + str(y)
      
      if analogy in seen : 
          continue 
        
      if y == 1 : 
          positive = positive + 1
      else : 
          negative  = negative + 1
        
      if (positive > max_positive) and (negative > max_negative) :
          break
      if (positive > max_positive) and y == 1 :
          continue 
      if (negative > max_negative) and y == 0 :
          continue
      
      seen.add(analogy)
      out.write(analogy + "\n")
   

max_positive = max_negative = NB_ANALOGIES_VALID
positive = negative = 0

# For paraphrases
with open(out_analogies_valid, 'w+') as out :
    
    while True :
        random = data_para_train.sample(n=2) 
        y = int(int(random.iloc[0]['Quality']) == int(random.iloc[1]['Quality']) and int(random.iloc[1]['Quality']) == 1)

        analogy = str(random.iloc[0]['#1 ID']) + '|' + str(random.iloc[0]['#2 ID']) + '|' + str(random.iloc[1]['#1 ID']) + '|' + str(random.iloc[1]['#2 ID']) + '|' + str(y)
        
        if analogy in seen : 
            continue 
          
        if y == 1 : 
            positive = positive + 1
        else : 
            negative  = negative + 1
          
        if (positive > max_positive) and (negative > max_negative) :
            break
        if (positive > max_positive) and y == 1 :
            continue 
        if (negative > max_negative) and y == 0 :
            continue     
        
        seen.add(analogy)
        out.write(analogy + "\n")
        
        
seen = set()      
max_positive = max_negative = NB_ANALOGIES_TEST
positive = negative = 0

# For paraphrases
with open(out_analogies_test, 'w+') as out :
    
    while True :
        random = data_para_test.sample(n=2) 
        y = int(int(random.iloc[0]['Quality']) == int(random.iloc[1]['Quality']) and int(random.iloc[1]['Quality']) == 1)

        analogy = str(random.iloc[0]['#1 ID']) + '|' + str(random.iloc[0]['#2 ID']) + '|' + str(random.iloc[1]['#1 ID']) + '|' + str(random.iloc[1]['#2 ID']) + '|' + str(y)
        
        if analogy in seen : 
            continue 
          
        if y == 1 : 
            positive = positive + 1
        else : 
            negative  = negative + 1
          
        if (positive > max_positive) and (negative > max_negative) :
            break
        if (positive > max_positive) and y == 1 :
            continue 
        if (negative > max_negative) and y == 0 :
            continue     
        
        seen.add(analogy)
        out.write(analogy + "\n")