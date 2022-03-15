# Paper-Reading-list


# Pre-trained Language Models
+ **ELMo**: "Deep contextualized word representations". NAACL(2018) [[pdf]](https://arxiv.org/abs/1802.05365)
+ **GPT**: "Improving Language Understanding by Generative Pre-Training". [[pdf]](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf)
+ **Bert**: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". NAACL(2019) [[pdf]](https://arxiv.org/abs/1810.04805)
+ **RoBERTa**: "RoBERTa: A Robustly Optimized BERT Pretraining Approach". arXiv(2019) [[pdf]](https://arxiv.org/abs/1907.11692) [[note]](notes/RoBERTa.md)
+ **ALBERT**: "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations". ICLR(2020) [[pdf]](https://arxiv.org/abs/1909.11942) [[note]](notes/ALBERT.md)
+ **ELECTRA**: "ELECTRA: PRE-TRAINING TEXT ENCODERS AS DISCRIMINATORS RATHER THAN GENERATORS". ICLR(2020) [[pdf]](https://arxiv.org/abs/2003.10555) [[note]](notes/ELECTRA.md)
+ **BART**: "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension". ACL(2020) [[pdf]](https://arxiv.org/abs/1910.13461) [[note]](notes/BART.md)

# SLU
+ **RCSF**: "Cross-Domain Slot Filling as Machine Reading Comprehension". IJCAI(2021) [[pdf]](https://www.ijcai.org/proceedings/2021/0550.pdf) [[note]](notes/RCSF.md)
+ **QASF**: "QA-Driven Zero-shot Slot Filling with Weak Supervision Pretraining". ACL(2021) [[pdf]](https://aclanthology.org/2021.acl-short.83/) [[note]](notes/QASF.md)
+ **PCLC**: "Bridge to Target Domain by Prototypical Contrastive Learning and Label Confusion: Re-explore Zero-Shot Learning for Slot Filling". EMNLP(2021) [[pdf]](https://arxiv.org/abs/2110.03572) 在Coach的基础上加上了原型对比学习(PCL)，让slot value更接近label slot远离其他slot。还加上了label confusion(LC)方法建模source domain的slot和target domain的slot之间的约束关系。

# MRC
+ **TASE**: "A Simple and Effective Model for Answering Multi-span Questions". EMNLP(2020) [[pdf]](https://arxiv.org/abs/1909.13375) [[note]](https://zhuanlan.zhihu.com/p/461651200)

# NER
+ **TemplateNER**: "Template-Based Named Entity Recognition Using BART". ACL(2021) [[pdf]](https://arxiv.org/pdf/2106.01760.pdf) [[note]](https://zhuanlan.zhihu.com/p/462088365?)
+ **EntLM**: "Template-free Prompt Tuning for Few-shot NER". arXiv(2021) [[pdf]](https://arxiv.org/abs/2109.13532) [[note]](https://zhuanlan.zhihu.com/p/462458103)
+ **LightNER**: "LightNER: A Lightweight Generative Framework with Prompt-guided Attention for Low-resource NER". arXiv(2021) [[pdf]](https://arxiv.org/abs/2109.00720) [[note]](https://zhuanlan.zhihu.com/p/463356701)
+ **LEAR**: "Enhanced Language Representation with Label Knowledge for Span Extraction". EMNLP(2021) [[pdf]](https://arxiv.org/abs/2111.00884) [[note]](https://zhuanlan.zhihu.com/p/466735142)
+ **BERT-MRC**: "A Unified MRC Framework for Named Entity Recognition". ACL(2020) [[pdf]](https://arxiv.org/abs/1910.11476) 把NER问题建模成MRC问题，使用"Annotation guideline notes"(标注数据集的时候给标注人员看的指导书)作为question，通过question引入了label的知识，在flat-ner和nested-ner的多个数据集上取得了sota，其中nested-ner上提升很大。

# Others
