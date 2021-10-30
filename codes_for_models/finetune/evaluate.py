import torch
from tqdm import tqdm

def out_prob(model, tokenizer, dloader, ofile):
    results = []
    with torch.no_grad():
        for inputs in tqdm(list(dloader)):
            inputs = {k: v.cuda() for k, v in inputs.items() if k!='labels'}
            outputs = model(**inputs)
            probs = outputs.logits.softmax(dim = 1)
            prob_label_is_true = probs[:,-1]
            results.extend([str(i.cpu().numpy())+'\n' for i in prob_label_is_true])
    with open(ofile, 'w') as f:
        f.writelines(results)
