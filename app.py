import json

from flask import Flask, render_template, request, jsonify, session
from flask_pymongo import PyMongo
import cohere
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import re

app = Flask(__name__)


# Initialize Cohere API
co = cohere.Client('O1jRnF6ABMam4G5KRsMlKfH7qJDFypv13nLiPuMz')

# MongoDB Connection
app.config["MONGO_URI"] = "mongodb://localhost:27017/clinicalTrails"
mongo = PyMongo(app)

# Precompute trial embeddings (run this once or on app startup)
TRIAL_EMBEDDINGS = {}
TRIAL_DATA = list(mongo.db.trails.find({}, {"_id": 0}))




def precompute_trial_embeddings():
    global TRIAL_EMBEDDINGS, TRIAL_DATA
    batch_size = 10  # Process 10 trials per API call

    try :
        with open("trail_embedding.json","r") as f:
            TRIAL_EMBEDDINGS = json.load(f)
            f.close()
    except Exception as e:
        for i in range(0, len(TRIAL_DATA), batch_size):
            batch = TRIAL_DATA[i:i + batch_size]
            texts = []
            print("batch Processing ...." , i)
            for trial in batch:
                text = " ".join([
                    # " ".join(trial.get("protocolSection", {}).get("conditionsModule", {}).get("conditions", [])),
                    # trial.get("protocolSection", {}).get("eligibilityModule", {}).get("gender", ""),
                    # trial.get("protocolSection", {}).get("statusModule", {}).get("overallStatus", ""),
                    # " ".join([loc.get("city", "") + " " + loc.get("country", "") for loc in trial.get("protocolSection", {}).get("contactsLocationsModule", {}).get("locations", [])])
                    str(trial)
                ])
                texts.append(text if text.strip() else "unknown")  # Handle empty text

            try:
                embeddings = co.embed(texts=texts, model="embed-english-v3.0", input_type="classification").embeddings
                for j, embedding in enumerate(embeddings):
                    TRIAL_EMBEDDINGS[i + j] = embedding
            except cohere.errors.TooManyRequestsError:
                print("Rate limit hit, waiting 60 seconds...")
                time.sleep(60)  # Wait 1 minute to reset rate limit
                embeddings = co.embed(texts=texts, model="embed-english-v3.0", input_type="search_document").embeddings
                for j, embedding in enumerate(embeddings):
                    TRIAL_EMBEDDINGS[i + j] = embedding

            time.sleep(1.5)  # ~40 calls/minute = 1 call every 1.5 seconds
        print(f"Precomputed embeddings for {len(TRIAL_DATA)} trials")


precompute_trial_embeddings()
# with open("trail_embedding.json", "w+") as f:
#     f.write(json.dumps(TRIAL_EMBEDDINGS))
#     f.close()


user_info={}
user_session ={"session_data":{}}
# Routes
@app.route('/')
def chat():
    return render_template('chat.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    global user_info
    """Handles user messages and filters trials in real-time."""
    data = request.json
    user_response = data.get("message")
    # if "condition" not in user_info:
    #     user_info["condition"]=1
    #     return jsonify({"response": "What specific condition are you looking for (e.g., heart disease, cancer)?"})

    # Initialize or load session
    if user_session["session_data"] == {}:
        user_session["session_data"] = {
            "previous_responses": [],
            "remaining_trials": list(range(len(TRIAL_DATA)))
        }
    session_data = user_session["session_data"]

    previous_responses = session_data["previous_responses"]
    remaining_trials = session_data["remaining_trials"]
    previous_responses.append(user_response)

    # Filter trials in real-time
    filtered_trials = filter_trials(remaining_trials, user_response, previous_responses)

    # Update session
    session_data["remaining_trials"] = filtered_trials
    session_data["previous_responses"] = previous_responses
    user_session["session_data"] = session_data # Ensure session updates

    # Check termination conditions
    if len(filtered_trials) == 1:
        trial = TRIAL_DATA[filtered_trials[0]]
        location = trial.get("protocolSection", {}).get("contactsLocationsModule", {}).get("locations", [])
        user_session["session_data"]= {}
        user_info={}
        return jsonify({
            "response": "Best matching trial found!",
            "geoPoints": [[location[0].get("geoPoint", {}).get("lat", 0), location[0].get("geoPoint", {}).get("lon", 0)]],
            "facilities": [location[0]]
        })

    if not filtered_trials:
        user_session["session_data"]= {}
        user_info = {}
        return jsonify({"response": "No matching trials found. Please restart the conversation."})

    # Generate next question
    next_question = generate_sub_questions(filtered_trials, previous_responses)
    return jsonify({"response": next_question})

def filter_trials(remaining_trials, user_response, previous_responses):
    """Filters trials using semantic similarity and all original filters."""
    # Embed user response
    full_context = " ".join(previous_responses)
    # Add input_type="search_query" for user input
    user_embedding = co.embed(texts=[full_context], model="embed-english-v3.0", input_type="classification").embeddings[0]
    user_embedding = np.array(user_embedding).reshape(1, -1)
    filtered_indices = []
    for idx in remaining_trials:
        trial = TRIAL_DATA[idx]
        trial_embedding = np.array(TRIAL_EMBEDDINGS[str(idx)]).reshape(1, -1)
        similarity = cosine_similarity(user_embedding, trial_embedding)
        # Get trial condition text for fuzzy matching fallback

        # Check similarity or fuzzy match for spelling tolerance
        # fuzzy_match = fuzz.partial_ratio(user_response.lower(), trial_conditions) > 70
        # Apply semantic threshold and all original filters
        if similarity[0][0] > 0.40 :
            filtered_indices.append(idx)

    return filtered_indices

def generate_sub_questions(filtered_trials, previous_responses):
    """Generate a question based on missing user info or trial data."""
    global user_info
    trials = [TRIAL_DATA[i] for i in filtered_trials[:]]  # Limit to first 5 for brevity

    # Prioritize missing key info
    #
    # if "age" not in user_info:
    #     user_info["age"]=1
    #     return "How old are you?"
    # if "location" not in user_info:
    #     user_info["location"]=1
    #     return "Where are you located (e.g., city or country)?"

    # If all key info is provided, ask a refining question based on trials
    context = f"""
    You are an intelligent clinical medical chatbot that helps users find the most relevant clinical trials.  
    Your goal is to **ask step-by-step filtering questions** to narrow down the available trials based on the user's responses.  
    
    Behavior:  
    1. **Acknowledge the user’s request** and provide the total number of matching trials.  
    2. **Ask one relevant filtering question at a time** to refine the results.  
    3. **Ensure each question is unique and not repeated** based on previous responses.  
    4. if the trails count for every 2 times is same   then ask the question  that diffferiate the trails 
    5. **Continue refining** until the list is small enough to present the final trials.  
    
    Example Flow:  
    User: "I am looking for clinical trials for lung cancer treatment."  
    Chatbot: "Got it! There are 120 trials currently available. Let's narrow it down. Are you looking for trials for yourself or someone else?"  
    
    User: "For myself."  
    Chatbot: "Thanks. Based on that, we now have 85 trials that might match your case. Could you please share your age?"  
    
    User: "I am 62 years old."  
    Chatbot: "Noted. That brings us to 40 available trials. Are you currently receiving any treatment for lung cancer, such as chemotherapy or immunotherapy?"  
    
    User: "I just finished chemotherapy last month."  
    Chatbot: "Understood. Now we have 12 trials that match your current treatment stage. Would you be willing to travel for a clinical trial, or do you prefer locations nearby?"  
    
    User: "I prefer trials within 100 miles of my location."  
    Chatbot: "Got it. Now we have 5 clinical trials available within 100 miles. Here’s a list of trials that match your criteria:"  
    
    Task:  
    - Given the remaining **{trials}** and the user's previous responses **{previous_responses}**, generate the next best **filtering question** that has not been asked yet.  
    - The question should **help refine** the results further in a natural, engaging, and structured way.  
    - **Just return the question.**  


    """
    try:
        response = co.generate(model="command", prompt=context, max_tokens=50, truncate="START")
        return response.generations[0].text.strip()
    except cohere.errors.TooManyRequestsError:
        print("Rate limit hit, waiting 60 seconds...")
        time.sleep(60)
        response = co.generate(model="command", prompt=context, max_tokens=50, truncate="START")
        return response.generations[0].text.strip()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
