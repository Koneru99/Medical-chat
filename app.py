

import json
import openai
import numpy as np
import time

import thefuzz.fuzz
from flask import Flask, render_template, request, jsonify
from flask_pymongo import PyMongo
from sklearn.metrics.pairwise import cosine_similarity
import thefuzz as fuzz
app = Flask(__name__)

# OpenAI API Key
openai.api_key = "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"
# MongoDB Connection
app.config["MONGO_URI"] = "mongodb://localhost:27017/clinicalTrails"
mongo = PyMongo(app)
client = openai.OpenAI(api_key="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")
# Precompute trial embeddings
TRIAL_EMBEDDINGS = {}
TRIAL_DATA = list(mongo.db.trails.find({}, {"_id": 0}))

def get_openai_embedding(text):
    """Get embeddings from OpenAI."""
    if len(text) > 8192:
        text = text[:8192]  # Truncate to fit model limit
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding


def getEligibilityCriteriaForBatch(trail_batch):
    eligibility_criteria = []
    for trail in trail_batch:
        eligibility = trail.get("protocolSection", {}).get("eligibilityModule", {})
        eligibility_criteria.append(eligibility.get("eligibilityCriteria", ""))
    return eligibility_criteria
def precompute_trial_embeddings():
    global TRIAL_EMBEDDINGS, TRIAL_DATA
    batch_size = 50  # Process 10 trials per API call

    try:
        with open("trial_embedding_eligibility.json", "r") as f:
            TRIAL_EMBEDDINGS = json.load(f)
    except Exception:
        for i in range(0, len(TRIAL_DATA), batch_size):
            batch = getEligibilityCriteriaForBatch(TRIAL_DATA[i:i + batch_size])
            texts = [str(trial) for trial in batch]
            print("Batch "+str(i)+ " is processing .......")
            try:
                embeddings = [get_openai_embedding(text) for text in texts]
                for j, embedding in enumerate(embeddings):
                    TRIAL_EMBEDDINGS[i + j] = embedding
            except openai.RateLimitError:
                print("Rate limit hit, waiting 60 seconds...")
                time.sleep(60)
                embeddings = [get_openai_embedding(text) for text in texts]
                for j, embedding in enumerate(embeddings):
                    TRIAL_EMBEDDINGS[i + j] = embedding

            time.sleep(1.5)
        print(f"Precomputed embeddings for {len(TRIAL_DATA)} trials")

precompute_trial_embeddings()
# with open("trial_embedding_eligibility.json", "w+") as f:
#     f.write(json.dumps(TRIAL_EMBEDDINGS))
#     f.close()
user_session = {"session_data": {}}
user_info =[]
user_set_questions={}
next_question=""
@app.route('/')
def chat():
    return render_template('chat.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    global user_info,user_set_questions,next_question
    """Handles user messages and filters trials in real-time."""
    data = request.json
    user_response = data.get("message")
    if "condition" not in user_info:
        user_info.append("condition")
        next_question="What specific condition are you looking for (e.g., heart disease, Covid 19,Breast Cancer)?"
        return jsonify({"response":next_question })

    if user_session["session_data"] == {}:
        user_session["session_data"] = {
            "previous_responses": [],
            "remaining_trials": list(range(len(TRIAL_DATA)))
        }
    session_data = user_session["session_data"]
    user_response = str(user_response).lower()
    previous_responses = session_data["previous_responses"]
    remaining_trials = session_data["remaining_trials"]
    previous_responses.append(user_response)
    filtered_trials = []
    if len(previous_responses)==1:
        filtered_trials = getDiseaseTrails(previous_responses[0])
    else:
        filtered_trials = filter_trials(remaining_trials, user_response, previous_responses)

    session_data["remaining_trials"] = filtered_trials
    session_data["previous_responses"] = previous_responses
    user_session["session_data"] = session_data

    if len(filtered_trials) == 1:
        trial = TRIAL_DATA[filtered_trials[0]]
        location = trial.get("protocolSection", {}).get("contactsLocationsModule", {}).get("locations", [])
        user_session["session_data"] = {}
        user_info=[]
        return jsonify({
            "response": "Best matching trial found!",
            "geoPoints": [[location[0].get("geoPoint", {}).get("lat", 0), location[0].get("geoPoint", {}).get("lon", 0)]],
            "facilities": [location[0]]
        })

    if not filtered_trials:
        user_info=[]
        user_session["session_data"] = {}
        return jsonify({"response": "No matching trials found. Please restart the conversation."})
    user_set_questions[next_question] = user_response
    next_question = generate_sub_questions(filtered_trials, user_set_questions)
    locations =[]
    for i in filtered_trials:
        trial = TRIAL_DATA[i]
        location = trial.get("protocolSection", {}).get("contactsLocationsModule", {}).get("locations", [])
        locations.append([location[0].get("geoPoint", {}).get("lat", 0), location[0].get("geoPoint", {}).get("lon", 0)])

    return jsonify({"response": next_question,"geoPoints": locations})

def getDiseaseTrails(disease):

    listitems = []
    disease_words = set(disease.lower().split())  # Convert disease to lowercase and split into words

    for i in range(len(TRIAL_DATA)):
        conditions = TRIAL_DATA[i].get("protocolSection",{}).get("conditionsModule",{}).get("conditions",[])
        for j in conditions:
            condition_words = set(j.lower().split())  # Convert condition to lowercase and split into words

            # Check for a fuzzy match (threshold 80%)
            if any(fuzz.fuzz.ratio(word1, word2) >= 60 for word1 in disease_words for word2 in condition_words):
                print(disease, j)
                listitems.append(i)
                break  # Exit inner loop once a match is found

    return listitems


def filter_trials(remaining_trials, user_response, previous_responses):
    """Filters trials using semantic similarity."""
    # Get user embedding (assuming get_openai_embedding is available)
    user_embedding = np.array(get_openai_embedding(previous_responses)).reshape(1, -1)

    # Compute cosine similarities
    similarities = [
        (idx, cosine_similarity(user_embedding, np.array(TRIAL_EMBEDDINGS[str(idx)]).reshape(1, -1))[0][0])
        for idx in remaining_trials
    ]

    # Extract raw similarity scores
    scores = np.array([sim for _, sim in similarities])

    # Normalize scores to [0, 1] range for consistency
    if scores.max() != scores.min():
        normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        normalized_scores = np.ones_like(scores) * 0.5  # Default to neutral if all scores are identical

    scaling_factor = 20
    probabilities = 1 / (1 + np.exp(-scaling_factor * (normalized_scores - 0.5)))

    # Sort by probability (descending)
    trial_probs = [(idx, prob) for (idx, _), prob in zip(similarities, probabilities)]
    trial_probs.sort(key=lambda x: x[1], reverse=True)

    mean_score = np.mean(probabilities)
    std_dev = np.std(probabilities)

    # Use mean + std deviation for a dynamic cutoff
    threshold = min(0.8, mean_score + (0.5 * std_dev))

    filtered_trials = [idx for idx, prob in trial_probs if prob >= threshold]

    return filtered_trials

def getElibilityCriteria(trail_data):
    eligibility = trail_data.get("protocolSection", {}).get("eligibilityModule", {})
    inclusion_criteria = eligibility.get("eligibilityCriteria", "")
    return inclusion_criteria

def generate_sub_questions(filtered_trials, previous_responses):
    """Generates next filtering question using OpenAI GPT."""
    trials = [getElibilityCriteria(TRIAL_DATA[i]) for i in filtered_trials][:9]
    context = f"""
        "You are a medical chatbot helping users find clinical trials. "
        "Given the list of eligibility criteria: \n" + {trials} + "\n"
        "Ask a specific, relevant filtering question based on the eligibility criteria "
        "that has not been asked before. Keep it concise. \n"
        f"Trial Count: {len(filtered_trials)}\n"
        f"Past Questions: {previous_responses}\n"
        "Return only the next question with also mentioning the Trail Count like this no of trails are remaining  in the question  {len(filtered_trials)}."
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": context}]
        )
        return response.choices[0].message.content.strip()
    except Exception as exception:
        print("Rate limit hit, waiting 60 seconds...")
        print(exception)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": context}]
        )
        return response.choices[0].message.content.strip()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
