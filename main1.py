from pyth_imports import *
from STRUCTS.dataset import *
from STRUCTS.models import *
from utils import *
  
tdebut = time.perf_counter()

config_loc = "config.txt"
config = pd.read_csv(config_loc, sep="|", index_col=False)

NB_ANALOGIES_TRAIN = int(config['NB_TRAIN'])
NB_ANALOGIES_VALID = int(config['NB_VALID'])


TRAIN_BATCH_SIZE  = int(config['BATCH_TRAIN'])
VALID_BATCH_SIZE  = int(config['BATCH_VALID'])

LR = float(config['LR'])
EPOCHS = int(config['EPOCHS'])
ENCODER = config['ENCODER'].to_string()
CLASSIFIER = config['CLASSIFIER'].to_string()
WEIGHTS = int(config['WEIGHTS'])

tokenizer = None
SENTENCE_TRANSFORMER = False
if ENCODER.__contains__('SBert') :
    encoder_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    #tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2')
    ENCODER_NAME = 'SBert'
    EMBEDDING_SIZE = 384
    SENTENCE_TRANSFORMER = True
elif ENCODER.__contains__('SroBERTa') :
    encoder_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    ENCODER_NAME = 'SroBERTa'
    SENTENCE_TRANSFORMER = True
elif ENCODER.__contains__('USE') :
    encoder_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    ENCODER_NAME = 'USE'
    SENTENCE_TRANSFORMER = True
elif ENCODER.__contains__('BERT-AVG') :
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoder_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    ENCODER_NAME = 'BERT-AVG'
elif ENCODER.__contains__('roBERTa-AVG') :
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')
    ENCODER_NAME = 'roBERTa-AVG'
elif ENCODER.__contains__('GloVe-AVG') :
    encoder_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    ENCODER_NAME = 'GloVe-AVG'
elif ENCODER.__contains__('word2vec-AVG') :
    encoder_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    ENCODER_NAME = 'word2vec-AVG'
else:
    print("Erreur dans le choix de l'encodeur\n Encodeurs disponibles : SBert")


if CLASSIFIER.__contains__('CNN'):
    model = Net()
    CLASSIFIER_NAME = 'CNN'
elif CLASSIFIER.__contains__('MLP'):
    model = MLP()
    CLASSIFIER_NAME = 'MLP'
else:
    print("Erreur dans le choix du Classifieur\n Classifieurs disponibles : CNN, MLP")


with open(results_loc, 'w+') as out :
    out.write('PARAMETRES EXPERIENCE : \nENCODER : ' + ENCODER_NAME +  '\t CLASSIFIER  : ' + CLASSIFIER_NAME + '\nLEARNING RATE : ' + str(LR) + '\t EPOCHS : ' + str(EPOCHS) +'\n\n')

       
t_debut_dataset = time.perf_counter()

training_set = AnalogiesData(encoder_model, 'msr-paraphrase-corpus/analogies_paraphrases_train_20.csv', 'msr-paraphrase-corpus/msr_paraphrase_data.txt', tokenizer)
valid_set = AnalogiesData(encoder_model, 'msr-paraphrase-corpus/analogies_paraphrases_valid_20.csv', 'msr-paraphrase-corpus/msr_paraphrase_data.txt', tokenizer)
testing_set = AnalogiesData(encoder_model, 'msr-paraphrase-corpus/analogies_paraphrases_test_20.csv', 'msr-paraphrase-corpus/msr_paraphrase_data.txt', tokenizer)

t_dataset = time.perf_counter()
print("Charglement des datasets fini en " + str(t_dataset - t_debut_dataset) + "s.")

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)


valid_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

valid_loader = DataLoader(valid_set, **valid_params)    


testing_loader = DataLoader(testing_set, **valid_params)    

            
		
model.to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

max_accu = 0
PATH = 'RESULTS/model.pt'
for epoch in range(EPOCHS):
    t1 = time.perf_counter()
    train(epoch, model, training_loader, criterion, optimizer)
    acc = valid(model, valid_loader, criterion, optimizer)
    t2 = time.perf_counter()

    if max_accu < acc:
        torch.save({
                'model_state_dict': model.state_dict(),
                }, PATH)
    else:
        model = model.load_state_dict(torch.load(PATH))
    with open(results_loc, 'a') as out :
        out.write("Epoch "+ str(epoch)+" training and validation performed in "+ str(t2 - t1) +" seconds\n")

    print("Epoch "+ str(epoch)+" training and validation performed in "+ str(t2 - t1) +" seconds\n")


acc = valid(model, testing_loader, criterion, optimizer, True)

tfin = time.perf_counter()
with open(results_loc, 'a') as out :
        out.write("Total running time : "+ str(tfin - tdebut) +" seconds\n")
print("Total running time : "+ str(tfin - tdebut) +" seconds\n")