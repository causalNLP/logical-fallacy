## How to use the pertrained models

The The model follows Huggingface standard format, for how to use models， you can refer the code in the file `codes_for_analysis/visualization/SaliencyMap_NLI.ipynb`.

Here is a simple example:

```python
import transformers
from transformers import pipeline
NLIer = pipeline("text-classification", model='the_path_to_the_model')

model = ModelWapper.model.cuda()
tokenizer = ModelWapper.tokenizer
```

