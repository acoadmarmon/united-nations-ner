# United Nations Named Entity Recognition
In many organizations, there is a unique vocabulary that maps names to known entities within that domain. At the United Nations, for instance, we have many specific entities which it is useful to identify in documents, including specific named committees and assemblies, important topics like the Sustainable Development Goals (SGDs), and many different countries and cultural groups that must be identified correctly. Exhaustively naming each and every important topic that may appear in a document, however, is not reasonable considering the shear number that may be important, especially considering the context of a document or sentence in which this entity is present. Instead, we want to be able to automatically identify and predict named entities using Named Entity Recognition (NER) to avoid creating a Regex parser with thousands or hundreds of thousands of possible named entities to match.

In this repository we use the Hugging Face library to fine-tune a BERT model on a new dataset to achieve better results on a domain specific NER task. In this case, we want to predict United Nations named entities in transcripts of meetings of the General Assembly. Key phrases such as "the general assembly", "second fifth committee", "the rohingya muslims", and more will need to be identified and extracted as named entities within these transcripts in a consistent and automated fashion.

[*See the full write-up on Medium.*](https://medium.com/p/d51d4cb3d7b5/edit)

## Predict Entities on New Text

Install the necessary dependencies with pip.

`pip install -r requirements.txt`

Next, unzip the un-ner.model.zip folder to use the fine-tuned weights. 

`unzip un-ner.model.zip`

Otherwise, you can fine-tune the model with your own paramters as shown below.

Then open the predict.ipynb jupyter notebook and replace the text variable with one of your choosing. This notebook is set up to load the un-ner.model weights, tokenize your input, and predict using the fine-tuned model. The results will be output to un_ner.csv.


## Fine-Tune the BERT NER Model

Install the necessary dependencies with pip.

`pip install -r requirements.txt`

Then simply run the train_un_ner.py file.

`python ./train_un_ner.py`

This will train, evaluate, and then save the fine-tuned model weights to un-ner.model which can be used to predict on new text.


