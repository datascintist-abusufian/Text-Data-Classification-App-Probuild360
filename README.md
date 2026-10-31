# Text Classification App (built for Probuild360)

A Streamlit app that classifies short political statements by truthfulness. It uses a TF-IDF and Naive Bayes pipeline, with interactive exploration, batch prediction and visual explanations.

## Data

`test.csv` holds about 1,260 labelled statements with speaker, party, subject and context fields. This format matches the [LIAR dataset](https://aclanthology.org/P17-2067/) of PolitiFact statements (Wang, 2017).

## Interface

The app has six tabs:

1. **Data Explorer:** class balance, word frequencies and word clouds
2. **Single Prediction:** classify one statement and see the class probabilities
3. **Model Analysis:** accuracy, classification report, confusion matrix and t-SNE view
4. **Batch Processing:** classify an uploaded CSV
5. **History:** past predictions in the session
6. **Advanced Analytics:** extra views of the data and model

The GIFs (`Bayes.gif`, `PCA_GIF.gif`, `markov_chain.gif`) illustrate Naive Bayes, PCA and Markov-chain text generation in the app.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Licence

No licence has been declared yet.

## Author

Md Abu Sufian - ORCID [0009-0007-3503-6942](https://orcid.org/0009-0007-3503-6942)
