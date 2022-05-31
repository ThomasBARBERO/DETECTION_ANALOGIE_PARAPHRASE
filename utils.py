from pyth_imports import *
from STRUCTS.models import *

def calculate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()
    return n_correct


def calculate_wrong(preds, targets):
    n_correct = (preds!=targets).sum().item()
    return n_wrong
	
def train(epoch, model, training_loader, criterion, optimizer):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0

    model.train()
  
    for i, data in enumerate(training_loader, 0):
      inputs =  data['data'].to(device)
      targets =  data['targets'].to(device)

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = model(inputs)
      loss = criterion(outputs, targets)

      
      tr_loss += loss.item()
      big_val, big_idx = torch.max(outputs.data, dim=1)
      n_correct += calculate_accuracy(big_idx, targets)

      nb_tr_steps += 1
      nb_tr_examples+=targets.size(0)
      
      if i%5000==0:
          loss_step = tr_loss/nb_tr_steps
          accu_step = (n_correct*100)/nb_tr_examples 
          #print(f"Training Loss per 5000 steps: {loss_step}")
          #print(f"Training Accuracy per 5000 steps: {accu_step}")

      loss.backward()
      optimizer.step()

    """ print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")"""

    return
	
	
def valid(model, testing_loader, criterion, optimizer, test=False):
    model.eval()
    n_correct = 0; n_wrong = 0; total = 0; tr_loss=0; nb_tr_steps=0; nb_tr_examples=0
    f1_scores = []
    precisions = []
    recalls  = []

    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            inputs =  data['data'].to(device)
            targets =  data['targets'].to(device)

            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calculate_accuracy(big_idx, targets)
            #n_wrong += calculate_wrong(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)
            f1_scores.append(f1_score(big_idx, targets, zero_division=1))
            precisions = [precision_score(big_idx, targets, zero_division=1)]
            recalls  = [recall_score(big_idx, targets, zero_division=1)]

            if _%5000==0:
                loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct*100)/nb_tr_examples
                #print(f"Validation Loss per 100 steps: {loss_step}")
                #print(f"Validation Accuracy per 100 steps: {accu_step}")

    #epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples


    with open(results_loc, 'a') as out :
        if (test):
            out.write("\n\Test report :\n")
        else:
            out.write("\n\Valid epoch report :\n")
        out.write("Average f1 score : " + str(np.mean(f1_scores)) + '\n')
        out.write("Average precision : " + str(np.mean(precisions)) + '\n')
        out.write("Average recall : " + str(np.mean(recalls)) + '\n')
        out.write("Epoch accuracy : " + str(epoch_accu) + '\n')
        out.write("--END--\n")
    
    return epoch_accu

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask