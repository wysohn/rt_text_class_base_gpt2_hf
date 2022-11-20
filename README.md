GPT2 Huggingface Transformer for text (document/sentence) classification, implemented as per Ready Tensor Text Classification - Base specifications.

- text classification
- huggingface
- gpt2
- nlp
- sklearn
- python
- pandas
- numpy
- flask
- nginx
- gunicorn
- docker

This is a Text Classifier that uses a GPT2ForSequenceClassification transformer model implemented through HuggingFace.

The programming language used for the model is Python, and the model is provided as service through Flast + Nginx + gunicorn, which the user can access it through the endpoints: `/ping` to check if the service is available, and `/infer` to request for prediction. The prediction accepts csv files (meaning that the HTTP header includes `Content-type: text/csv`), and no other formats are accepted at the moment.

Preprocessing step includes fitting the LabelEncoder() of sklearn.preprocessing package using the training dataset's label and the pre-trained tokenizer, GPT2Tokenizer, from HuggingFace.

The model is ready to accept any dataset as specified by Ready Tensor Text Classification - Base specification, and the model is already pre-validated to be some of the example datasets, such as clickbait, drug, and movie reviews as well as spam texts and tweets from Twitter.