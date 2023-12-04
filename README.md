# Logical Fallacy Detection
Repo for the paper "[Logical Fallacy Detection
](https://arxiv.org/abs/2202.13758)" (Findings of EMNLP 2022) by _Abhinav Lalwani*, Zhijing Jin*, Tejas Vaidhya, Xiaoyu Shen, Yiwen Ding, Zhiheng Lyu, Mrinmaya Sachan, Rada Mihalcea, Bernhard Schoelkopf_.

#### Dataset
Feel free to access our data in the `data/` folder. 

Currently, all the data are annotated in the sentence format. In addition, if you need detailed source of our climate data, especially in the form of original articles, we have them collected in [this spreadsheet](https://docs.google.com/spreadsheets/d/1AHyjRkXk4xEmRWGc-k07LA1lflZQK9UIOw921zXZLqA/edit#gid=0).

#### To cite the paper:
```bibtex
@inproceedings{jin-etal-2022-logical,
    title = "Logical Fallacy Detection",
    author = "Jin, Zhijing  and
      Lalwani, Abhinav  and
      Vaidhya, Tejas  and
      Shen, Xiaoyu  and
      Ding, Yiwen  and
      Lyu, Zhiheng  and
      Sachan, Mrinmaya  and
      Mihalcea, Rada  and
      Schoelkopf, Bernhard",
    editor = "Goldberg, Yoav  and
      Kozareva, Zornitsa  and
      Zhang, Yue",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.532",
    doi = "10.18653/v1/2022.findings-emnlp.532",
    pages = "7180--7198",
    abstract = "Reasoning is central to human intelligence. However, fallacious arguments are common, and some exacerbate problems such as spreading misinformation about climate change. In this paper, we propose the task of logical fallacy detection, and provide a new dataset (Logic) of logical fallacies generally found in text, together with an additional challenge set for detecting logical fallacies in climate change claims (LogicClimate). Detecting logical fallacies is a hard problem as the model must understand the underlying logical structure of the argument. We find that existing pretrained large language models perform poorly on this task. In contrast, we show that a simple structure-aware classifier outperforms the best language model by 5.46{\%} F1 scores on Logic and 4.51{\%} on LogicClimate. We encourage future work to explore this task since (a) it can serve as a new reasoning challenge for language models, and (b) it can have potential applications in tackling the spread of misinformation. Our dataset and code are available at \url{https://github.com/causalNLP/logical-fallacy}",
}
```

#### Contact:
- For future collaboration requests, feel free to email Zhijing Jin and Mrinmaya Sachan.
- For coding questions, feel free to email Abhinav Lalwani.
- For dataset collection-related questions,feel free to email Abhinav Lalwani, Zhijing Jin, and Yiwen Ding.
