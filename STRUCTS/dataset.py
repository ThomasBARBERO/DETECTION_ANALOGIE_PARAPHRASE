from pyth_imports import *
from utils import *

class AnalogiesData(Dataset):
    def __init__(self, encoder, filename_analogies, filename_phrases, tokenizer = None):
      self.encoder = encoder.to(device)
      self.tokenizer = tokenizer
      t1 = time.perf_counter()
      self.load_from_csv(filename_analogies, filename_phrases)
      t2 = time.perf_counter()
      #print(str(t2 - t1))
      self.encode()
      t3 = time.perf_counter()
      t_total = t3 - t1
      #print("Temps total de l'initialisation du dataset : " + str(t_total) +" | Temps chargement depuis CSV : " + str((t2 -t1)/t_total * 100) + " | Temps encodage : " + str((t3 -t2)/t_total * 100))
    
    def load_from_csv(self, location_analogies, location_phrases):
      self.analogies_ID = pd.read_csv(location_analogies, names=['A', 'B', 'C', 'D', 'y'], sep='|')
      self.phrases = pd.read_csv(location_phrases, names=['ID',	'String',	'Author',	'URL',	'Agency',	'Date',	'WebDate'], sep='\t')
      self.phrases.drop(index=self.phrases.index[0], axis=0, inplace=True)
      self.phrases = self.phrases.astype({"ID": int}, errors='raise') 


      self.dic_phrasesID = dict(zip(self.phrases.ID, self.phrases.String))
      a = self.analogies_ID.A.map(self.dic_phrasesID)
      b = self.analogies_ID.B.map(self.dic_phrasesID)
      c = self.analogies_ID.C.map(self.dic_phrasesID)
      d = self.analogies_ID.D.map(self.dic_phrasesID)
      data_temp = {'A':a, 'B':b, 'C':c, 'D':d, 'y':self.analogies_ID.y}
      self.data = pd.DataFrame(data=data_temp)


    def encode(self):
        t1 = time.perf_counter()
        if (self.tokenizer == None):
            t5 = time.perf_counter()
            temp = self.data.A
            t6 = time.perf_counter()
            self.A = self.encoder.encode(temp)
            t7 = time.perf_counter()
            t_total = t7 - t5
            print("Temps total : " + str(t_total) + " | Temps 1 : "  + str((t6 -t5)/t_total * 100) + " | Temps 2 : "  + str((t7 -t6)/t_total * 100))

            self.B = self.encoder.encode(self.data.B)
            self.C = self.encoder.encode(self.data.C)
            self.D = self.encoder.encode(self.data.D)
        else:
            a_ids = self.tokenizer(self.data.A.tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt')
            b_ids = self.tokenizer(self.data.B.tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt')
            c_ids = self.tokenizer(self.data.C.tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt')
            d_ids = self.tokenizer(self.data.D.tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt')
            a_encoded = self.encoder(**a_ids)
            b_encoded = self.encoder(**b_ids)
            c_encoded = self.encoder(**c_ids)
            d_encoded = self.encoder(**d_ids)
            self.A = mean_pooling(a_encoded, a_ids['attention_mask'])
            self.B = mean_pooling(b_encoded, b_ids['attention_mask'])
            self.C = mean_pooling(c_encoded, c_ids['attention_mask'])
            self.D = mean_pooling(d_encoded, d_ids['attention_mask'])
            
            print(type(self.A) + "aaaaaaaaaa")
           
        t2 = time.perf_counter()
        self.data = np.array([self.A,self.B,self.C,self.D])
        self.targets = self.analogies_ID.y.tolist()
        
        t3 = time.perf_counter()
        self.data = torch.Tensor(self.data)
        self.data = torch.transpose(self.data, 0, 1)
        self.data = torch.unsqueeze(self.data, 1)
        self.targets = torch.tensor(self.targets, dtype=torch.long)
        self.data.to(device)
        self.targets.to(device)
        t4 = time.perf_counter()

        t_total = t4 - t1
        #print("Temps total : " + str(t_total) + " | Temps 1 : "  + str((t2 -t1)/t_total * 100) + " | Temps 2 : "  + str((t3 -t2)/t_total * 100) + " | Temps 3 : "  + str((t4 -t3)/t_total * 100))

        return [self.data, self.targets]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        targets_ = self.targets[index].clone().detach()
        #targets.to(torch.long)
        #return {'data': self.data[index].clone().detach(), 'targets': torch.tensor(self.targets[index], dtype=torch.long) }
        return {'data': self.data[index].clone().detach(), 'targets': targets_ }