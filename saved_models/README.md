## How to use the pretrained models

Pretrained Models can be downloaded from this [link](https://drive.google.com/drive/folders/13icpZY_HNemv9Da14fYEyw8EfoxEvKQZ?usp=sharing)


The model follows Huggingface standard format, for how to use models， you can refer the code in the file `codes_for_models/experiments_round2/logicedu.py` , and use the appropriate flags

Here is an example on how to load the models:

```python
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model =  AutoModelForSequenceClassification.from_pretrained('path_to_saved_model', num_labels=3)
tokenizer = AutoTokenizer.from_pretrained('path_to_tokenizer', do_lower_case=True)



```

