Repository for paper "Behavior Alignment: A New Perspective of Evaluating LLM-based
Conversational Recommendation Systems" published at The 47th International ACM SIGIR Conference on Research and Development in Information Retrieval.

Update: Repo under meantainance, more content is to-be updated.



The checkpoint of Binary Classifier for computing intrinct Behavior Alignment can be found under huggingface model repos:
Original: `Dylan1999/Behavior_Alignment_Origin_Binary_Classifier`
Mixed-hard: `Dylan1999/Behavior_Alignment_MixedHard_Binary_Classifier`



There are 13 distinct behavior types in CRS defined in INSPIRED dataset. We replaced them to IDs, and the mapping table is:
```
    id2label = {
        0: "acknowledgment",
        1: "credibility",
        2: "encouragement",
        3: "experience_inquiry",
        4: "offer_help",
        5: "opinion_inquiry",
        6: "personal_experience",
        7: "personal_opinion",
        8: "preference_confirmation",
        9: "rephrase_preference",
        10: "self_modeling",
        11: "similarity",
        12: "transparency"  
    }

```


Under `evaluation` folder, you can find the code to output evaluation results under Section 5.2.

For OOD eval data, `Redial` dataset is considered as Out-of-distribution. Since our model is trained using data from INSPIRED dataset.

- `OOD_data_creation`: Code to create `redial_eval_dataset.json`
- `inference.py`: file to create two evaluation result `.csv` file under OOD_inference_result, which load the trained checkpoint and inference the predictions.