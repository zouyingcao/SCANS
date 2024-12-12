import json
import requests
import pandas as pd
from googleapiclient import discovery
from utils.modeling_utils import is_reject
from sklearn.metrics import precision_score, recall_score, f1_score

def llama_guard(texts):
    
    predicted_labels = []
    API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-Guard-2-8B"
    headers = {"Authorization": "Bearer hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()
    
    for text in texts: 
        output = query({
            "inputs": text,
        })
        print(output)

def perspective_api(texts):
    
    API_KEY = 'PERSPECTIVE_API_KEY'

    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )

    for text in texts:
        analyze_request = {
        'comment': { 'text': text },
        'requestedAttributes': {'TOXICITY': {}}
        }

        response = client.comments().analyze(body=analyze_request).execute()
        print(json.dumps(response, indent=2))
        


if __name__ == "__main__":
    xstest = pd.read_csv('datasets/xstest_v2_prompts.csv')
    harmful_texts = xstest[xstest['type'].str.contains('contrast_')]['prompt'].tolist()
    harmless_texts = xstest[~xstest['type'].str.contains('contrast_')]['prompt'].tolist()
    texts = harmful_texts.extend(harmless_texts)
    true_labels = [0]*len(harmful_texts)+[1]*len(harmless_texts)
    
    predicted_labels = llama_guard(texts)
    
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
     