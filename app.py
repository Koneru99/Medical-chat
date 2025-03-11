import time

from cohere.finetuning import FinetunedModel, Settings, BaseModel
from flask import Flask, render_template, request, jsonify
import cohere
import re

import trail_database
# from trail_database import getTheCountBasedOnConditions, read_data, extractTrailsBasedOnlocations,getTrialsBasedOnAge

app = Flask(__name__)

# Initialize Cohere API
co = cohere.Client('O1jRnF6ABMam4G5KRsMlKfH7qJDFypv13nLiPuMz')

# # Load the trial database
# trial_db = read_data()

# # Global variables for tracking conversation state
# current_question = 0
# user_responses = {}
# disease_found = False
# matching_trials = []

# # Predefined questions
# questions = [
#     "What is your age? (Please provide only a number)",
#     "Are you high-risk for contracting the Covid19/novel coronavirus infection (e.g., recent contact with infected person, healthcare worker)? (yes/no)",
#     "Have you been diagnosed with the novel coronavirus/COVID-19 infection? (yes/no)",
#     "If yes, Please provide the date of your positive coronavirus/COVID-19 test.",
#     "Are you currently hospitalized or in an intensive Care Unit (ICU)? (yes/no)",
#     "Are you currently pregnant? (yes/no)"
# ]

# # Utility functions
# def extract_date(input_text):
#     """Extracts a date from the input text using regex."""
#     date_pattern = r'\b(?:\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4})\b'
#     match = re.search(date_pattern, input_text)
#     return match.group(0) if match else None

# def extract_entity(input_text, entity_type):
#     """
#     Extract entities like disease or location using Cohere API.
#     """
#     prompt = f"""
#     Extract the {entity_type} from the following text:
#     Text: "I think I have heart disease."
#     {entity_type.capitalize()}: heart disease

#     Text: "I am located in Los Angeles."
#     {entity_type.capitalize()}: Los Angeles

#     Text: "I was diagnosed with diabetes."
#     {entity_type.capitalize()}: diabetes

#     Text: "{input_text}"
#     {entity_type.capitalize()}:"""
    
#     response = co.generate(
#         model="command-xlarge-nightly",
#         prompt=prompt,
#         max_tokens=10,
#         temperature=0,
#         stop_sequences=["\n"]
#     )
#     return response.generations[0].text.strip()

# # Routes
# @app.route('/')
# def chat():
#     return render_template('chat.html')

# @app.route('/send_message', methods=['POST'])
# def send_message():
#     """Handles user input, predefined questions, and dynamic responses."""
#     global current_question, user_responses, disease_found, matching_trials

#     user_input = request.json.get('message')
#     if current_question < len(questions):
#         # Handle predefined health-related questions
#         return handle_questions(user_input)
#     elif disease_found and  matching_trials:
#         # Handle location-specific questions
#         return handle_location_question(user_input)
#     else:
#         # Handle dynamic or unrelated queries
#         return handle_disease_or_dynamic(user_input)

# def handle_questions(user_input):
#     """Handles predefined questions in sequence."""
#     global current_question, user_responses

#     current_question_index = current_question

#     if current_question_index == 0:  # Age
#         try:
#             age = int(user_input)
#             user_responses['age'] = age
#             current_question += 1
#             return jsonify({"response": questions[1]}), 200
#         except ValueError:
#             return jsonify({"response": "Please provide your age as a number."}), 400

#     elif current_question_index == 1:  # High-risk for Covid-19
#         if user_input.lower() in ['yes', 'no']:
#             user_responses['high_risk'] = user_input.lower()
#             current_question += 1
#             return jsonify({"response": questions[2]}), 200
#         else:
#             return jsonify({"response": "Please answer with 'yes' or 'no'."}), 400

#     elif current_question_index == 2:  # Diagnosed with COVID-19
#         if user_input.lower() in ['yes', 'no']:
#             user_responses['diagnosed_covid'] = user_input.lower()
#             if user_input.lower() == 'yes':
#                 current_question += 1
#                 return jsonify({"response": questions[3]}), 200
#             else:
#                 current_question += 2
#                 return jsonify({"response": questions[4]}), 200
#         else:
#             return jsonify({"response": "Please answer with 'yes' or 'no'."}), 400

#     elif current_question_index == 3:  # COVID-19 test date
#         user_responses['covid_test_date'] = extract_date(user_input) or user_input
#         current_question += 1
#         return jsonify({"response": questions[4]}), 200

#     elif current_question_index == 4:  # Hospitalized
#         if user_input.lower() in ['yes', 'no']:
#             user_responses['hospitalized'] = user_input.lower()
#             current_question += 1
#             return jsonify({"response": questions[5]}), 200
#         else:
#             return jsonify({"response": "Please answer with 'yes' or 'no'."}), 400

#     elif current_question_index == 5:  # Pregnant
#         if user_input.lower() in ['yes', 'no']:
#             user_responses['pregnant'] = user_input.lower()
#             current_question += 1
#             return jsonify({"response": "Thank you for your responses! Please provide your concern."}), 200
#         else:
#             return jsonify({"response": "Please answer with 'yes' or 'no'."}), 400

# def handle_location_question(user_input):
#     """Handles location-specific questions after a disease is identified."""
#     global matching_trials, disease_found,current_question

#     print(user_input)

#     location = extract_entity(user_input, 'location')
#     if location:
#         filtered_locations = extractTrailsBasedOnlocations(matching_trials, location)
#         if filtered_locations:
#             response_data = {
#                 "response": f"There are {len(filtered_locations)} trials in {location}.",
#                 "geoPoints": [[i.geoPoint.lat, i.geoPoint.lon] for i in filtered_locations],
#                 "facilities": filtered_locations
#             }
#             matching_trials = []
#             disease_found = False
#             current_question=0     
#             return jsonify(response_data), 200
#         else:
#             return jsonify({"response": f"No trials found for {location}. Please provide another location."}), 200
#     else:
#         return jsonify({"response": "I couldn't extract your location. Please provide it again."}), 400

# def handle_disease_or_dynamic(user_input):
#     """Handles disease extraction or unrelated dynamic queries."""
#     global disease_found, matching_trials
#     matching_trials = getTrialsBasedOnAge(trial_db,user_responses["age"])
#     if not disease_found:
#         disease = extract_entity(user_input, 'disease')
#         if disease:
#             matching_trials = getTheCountBasedOnConditions(matching_trials, disease)
#             if matching_trials:
#                 disease_found = True
#                 return jsonify({"response": f"There are {len(matching_trials)} trials available for {disease}. Please provide your location."}), 200
#             else:
#                 return jsonify({"response": f"No trials found for {disease}."}), 200
#         else:
#             response = handle_dynamic_questions(user_input)
#             return jsonify({"response": response}), 200
#     else:
#         response = handle_dynamic_questions(user_input)
#         return jsonify({"response": response}), 200

# def handle_dynamic_questions(user_input):
#     """Handles unrelated dynamic queries."""
#     prompt = f"""
#     Respond to the following user query:
#     User: "{user_input}"
#     Bot:"""
    
#     response = co.generate(
#         model="command-xlarge-nightly",
#         prompt=prompt,
#         max_tokens=50,
#         temperature=0.7,
#         stop_sequences=["\n"]
#     )
#     return response.generations[0].text.strip()


# import faiss
# import numpy as np
# import json

# # Load embeddings
# with open("trial_vectors.json", "r") as f:
#     trial_vectors = json.load(f)

# # Convert to numpy array
# dimension = len(trial_vectors[0]["vector"])
# index = faiss.IndexFlatL2(dimension)
# vectors = np.array([t["vector"] for t in trial_vectors]).astype("float32")
# index.add(vectors)

# # Save index
# faiss.write_index(index, "trials.index")


# import faiss
# import numpy as np

# def retrieve_relevant_trials(query):
#     """Retrieve the most relevant trials based on user query."""
#     index = faiss.read_index("trials.index")
#     query_vector = np.array([embed_text(query)]).astype("float32")

#     distances, indices = index.search(query_vector, k=3)  # Get top 3 matches
#     with open("trial_vectors.json", "r") as f:
#         trials = json.load(f)
    
#     matched_trials = [trials[i]["text"] for i in indices[0]]
#     return matched_trials

# @app.route('/send_message', methods=['POST'])
# def handle_dynamic_questions(user_input):
#     """Retrieve relevant trials and generate chatbot responses."""
#     relevant_trials = retrieve_relevant_trials(user_input)
    
#     prompt = f"""
#     The user asked: "{user_input}"
#     Based on our trial database, here are relevant trials:
#     - {relevant_trials[0]}
#     - {relevant_trials[1]}
#     - {relevant_trials[2]}

#     Generate a structured, friendly response:
#     """

#     response = co.generate(
#         model="command",
#         prompt=prompt,
#         max_tokens=100,
#         temperature=0.7,
#         stop_sequences=["\n"]
#     )

#     return jsonify({"response": response.generations[0].text.strip()}), 200



import json
import cohere
from flask import Flask, request, jsonify

app = Flask(__name__)

# Initialize Cohere API
COHERE_API_KEY = "Kwf7nT6IwNFn6LzGDizdH5LLX8SlQAOMxUMpTy9E"  # Replace with your actual API key
cohere_client = cohere.Client(COHERE_API_KEY)

# Define the fine-tuning parameters
model = "command"
dataset_path = "formatted_data1.jsonl"
epochs = 3
learning_rate = 0.001
# # create a dataset

# my_dataset = cohere_client.datasets.create(
#     name="customer",
#     type="chat-finetune-input",
#     data=open(dataset_path, "rb")
# )

# result = co.wait(my_dataset)
# model_time = int(time.time())
# model_name = f"my_finetuned_model_{model_time}"
# # Fine-tune the model with your dataset
# finetuned_model = cohere_client.finetuning.create_finetuned_model(
#     request=FinetunedModel(
#         name=model_name,
#         settings=Settings(
#             base_model=BaseModel(
#                 base_type="BASE_TYPE_CHAT",
#             ),
#             dataset_id="formatted-data1-y0zyeb",
#         ),
#     ),
# ).finetuned_model
# model_status = cohere_client.finetuning.get_finetuned_model(finetuned_model.id).finetuned_model
# # Wait for model to be ready
# while True:
#     print(model_status.status)
#     if model_status.status in "STATUS_READY":
#         print(f"Model {finetuned_model.id} is ready!")
#         break
#     elif model_status.status in ( "STATUS_TEMPORARILY_OFFLINE","STATUS_FAILED","STATUS_DELETED"):
#         raise Exception(f"Fine-tuning failed: {model_status}")
#     else:
#         print(f"Waiting for model {finetuned_model.id} to be ready...")
#         time.sleep(30)  # Wait 30 seconds before checking again
#     #
#     #
#     #

model_id = "b54296fd-bba4-4e07-9dad-fa6e34cff3ff"
# Store user responses
user_sessions = {}

model_status = cohere_client.finetuning.get_finetuned_model(model_id).finetuned_model
print(model_status.status)


@app.route('/')
def chat():
    return render_template('chat.html')

@app.route("/send_message", methods=["POST"])
def chatmessage():
    user_input = request.json.get("message")
    print(user_input)
    response = cohere_client.generate(
        model=model_id+"-ft",
        prompt=user_input,
        max_tokens=50,
        temperature=0.7
    )
    # Print the model's response
    print("Model Response:", response.generations[0].text)

    # return response.generations[0].text.strip()
    return response

#
# def process_chat(user_input):
#     session_data = user_sessions
#     global trials_data
#
#     # Stop conversation if user enters 'stop'
#     if user_input.lower() == "stop":
#         session_data.clear()
#         return jsonify({"response": "Conversation stopped. Let me know if you need any help later!"})
#
#     # Check if the user is changing topics
#     if detect_unrelated_query(user_input):
#         return jsonify({"response": handle_unrelated_query(user_input)})
#
#     # If user is answering a specific question
#     if "last_filter" in session_data:
#
#         del session_data["last_filter"]
#
#     # Define step-by-step filtering order
#     conversation_steps = ["greeting","condition", "age", "current_treatment", "travel_preference"]
#     pending_filters = [f for f in conversation_steps if f not in session_data]
#
#     if pending_filters:
#         next_filter = pending_filters[0]
#         session_data["last_filter"] = next_filter
#         if next_filter == "age" and not user_input.isdigit():
#             return jsonify({"response": "Please enter a valid number for age."})
#         session_data[session_data["last_filter"]] = user_input.lower()
#         if session_data["last_filter"] == "age":
#             trials_data = getTrialsBasedOnAge(trials_data, int(user_input))
#         if session_data["last_filter"] == "condition":
#             trials_data = getTheCountBasedOnConditions(trials_data, user_input.lower())
#         if session_data["last_filter"] == "travel_preference":
#             trials_data =extractTrailsBasedOnlocations(trials_data, user_input)
#
#         return jsonify({"response": generate_response(next_filter,len(trials_data))})
#
#     # Apply all collected filters to dataset)
#     trial_count = len(trials_data)
#
#     if trial_count > 0:
#         response_data = {
#             "response": generate_response("results", trial_count),
#             "geoPoints": extract_geo_points(trials_data),
#             "facilities": trials_data
#         }
#         session_data.clear()
#         return jsonify(response_data)
#     else:
#         session_data.clear()
#         return jsonify({"response": "I’m sorry, but no matching clinical trials were found."})
#
#
# def generate_response(next_filter, trial_count=None):
#     prompts = {
#         "greeting": "Hello! I'm here to help you find a suitable clinical trial. Let's begin!",
#         "condition": "Got it! Could you please tell me the medical condition you're searching for a clinical trial for?",
#         "eligibility": lambda
#             count: f"There are {count} trials available. Are you looking for trials for yourself or someone else?",
#         "age": lambda
#             count: f"Thanks! Based on that, we now have {count} trials that might match. Could you please share your age? (Numbers only)",
#         "current_treatment": lambda
#             count: f"Noted. That brings us to {count} available trials. Are you currently receiving any treatment, such as chemotherapy or immunotherapy?",
#         "travel_preference": lambda
#             count: f"Understood. Now we have {count} trials that match your current treatment stage. Would you be willing to travel for a clinical trial, or do you prefer locations nearby?",
#         "results": lambda
#             count: f"Got it! Now we have {count} clinical trials available based on your criteria. Here’s a list of trials that match:",
#         "no_results": "I'm sorry, but I couldn’t find any matching trials. Would you like to adjust your search or explore other options?"
#     }
#
#     system_prompt = (
#         "You are a professional medical chatbot designed to assist users in finding clinical trials."
#         " Follow a structured conversation flow by first asking about the medical condition, then the user's eligibility,"
#         " age (numbers only), ongoing treatments, and travel preference."
#         " Always acknowledge the user's responses before moving to the next step and provide meaningful insights."
#         "provide the count in the question"
#     )
#     try:
#         response = co.generate(
#             model="command",
#             prompt=f"{system_prompt}\n\nUser: {prompts.get(next_filter, 'Can you clarify?')}\nBot:",
#             max_tokens=75,
#             temperature=0.3,
#         )
#         return response.generations[0].text.strip()
#     except cohere.errors.CohereError as e:
#         print(f"Error calling Cohere API: {e}")
#         return "I'm currently facing some issues generating a response. Please try again later."
#
#
# def detect_unrelated_query(user_input):
#     unrelated_keywords = ["explain", "what is", "define", "tell me about", "help", "stop", "exit"]
#     return any(keyword in user_input.lower() for keyword in unrelated_keywords)
#
#
# def handle_unrelated_query(user_input):
#     system_prompt = (
#         "You are a knowledgeable medical chatbot."
#         " If a user asks about a medical term or wants to pause their clinical trial search, provide a helpful response."
#         " Do not ask unrelated follow-up questions."
#     )
#
#     try:
#         response = co.generate(
#             model="command",
#             prompt=f"{system_prompt}\n\nUser: {user_input}\nBot:",
#             max_tokens=100,
#             temperature=0.5,
#         )
#         return response.generations[0].text.strip()
#     except cohere.errors.CohereError as e:
#         print(f"Error calling Cohere API: {e}")
#         return "I'm currently facing some issues generating a response. Please try again later."
#

def extract_geo_points(trials):
    return [[i.geoPoint.lat, i.geoPoint.lon] for i in trials]



if __name__ == "__main__":
    app.run(debug=True)


