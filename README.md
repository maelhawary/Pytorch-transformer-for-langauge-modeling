# Pytorch_Transformer_for_language_modeling
Pytorch implementation of a transformer-based deep learning model
for langauge_modeling. The model is built so you can use it for any other tasks that have the same sequence (e.g. time-series forcast, price prediction, etc..). The model in decoder only and it is slightly different than the vanella model in (ref). The difference is the post_LN and the Pre_LN ()...) and adding the norm inside the residual. In this project, the model is used to predict schekspeer note to understand the sche writing and the it can be used to imitate the note and generate continuous random notes similar to  schek based on what the model understand. 

Next project: TansGANs, reccurent Trans or herarchical Trans