import time

from cohere.finetuning import FinetunedModel, Settings, BaseModel
from flask import Flask, render_template, request, jsonify
import cohere
import re
from flask import Flask, jsonify, request
from flask_pymongo import PyMongo
from thefuzz import fuzz

app = Flask(__name__)

# Initialize Cohere API
co = cohere.Client('O1jRnF6ABMam4G5KRsMlKfH7qJDFypv13nLiPuMz')


# MongoDB Connection
app.config["MONGO_URI"] = "mongodb://localhost:27017/clinicalTrails"  # Replace 'mydatabase' with your database name
mongo = PyMongo(app)

user_session ={}
clinical_trails =[]
# Routes
@app.route('/')
def chat():
    return render_template('chat.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    global user_session
    """Handles user messages, filters trials dynamically, and generates the next question."""
    data = request.json
    user_response = data.get("message")
    # session = data.get("session")  # Stores previous responses + remaining trials

    # If this is a new session, fetch all trials
    print(user_response)
    if not user_session:
        user_session = {
            "previous_responses": [],
            "remaining_trials": list(mongo.db.trails.find({}, {"_id": 0}))
        }
    trials = user_session["remaining_trials"]
    previous_responses = user_session["previous_responses"]

    # Store user response for context
    previous_responses.append(user_response)

    # Dynamically filter trials based on the latest response
    filtered_trials = filter_trials(trials, user_response)

    # Update session
    user_session["remaining_trials"] = filtered_trials
    user_session["previous_responses"] = previous_responses

    # If only one trial remains, return it
    if len(filtered_trials) == 1:
        user_session.clear()
        location = filtered_trials[0].get("protocolSection", {}).get("contactsLocationsModule", {}).get("locations", [])
        return jsonify({"response": "Best matching trial found!",
                        "geoPoints": [[location[0].get("geoPoint",{}).get("lat",0),location[0].get("geoPoint",{}).get("lon",0)]] ,
                        "facilities": [location[0]]})

    # If no trials remain, restart the process
    if not filtered_trials:
        user_session.clear()
        return jsonify({"response": "No matching trials found. Please restart the conversation."})

    # Generate the next question using Cohere based on updated trials + past responses
    next_question = generate_sub_questions(filtered_trials, previous_responses)

    return jsonify({"response": next_question, "session": user_session})



from rapidfuzz import fuzz

def filter_trials(trials, user_response):
    """Filters clinical trials dynamically based on multiple conditions from the dataset."""
    filtered_trials = []

    for trial in trials:
        condition_match = match_condition(trial, user_response)
        age_match = check_age_eligibility(trial, user_response)
        gender_match = check_gender_eligibility(trial, user_response)
        health_status_match = check_health_status_eligibility(trial, user_response)
        location_match = check_location(trial, user_response)
        recruitment_match = check_recruitment_status(trial, user_response)
        intervention_match = check_intervention(trial, user_response)
        study_type_match = check_study_type(trial, user_response)
        sponsor_match = check_sponsor(trial, user_response)
        expanded_access_match = check_expanded_access(trial, user_response)
        genetic_study_match = check_genetic_study(trial, user_response)
        sample_size_match = check_sample_size(trial, user_response)
        fda_regulated_match = check_fda_regulation(trial, user_response)

        if any([
            condition_match, age_match, gender_match, health_status_match, location_match, recruitment_match,
            intervention_match, study_type_match, sponsor_match, expanded_access_match, genetic_study_match,
            sample_size_match, fda_regulated_match
        ]):
            filtered_trials.append(trial)

    return filtered_trials


def match_condition(trial, user_response):
    """Fuzzy match user input with trial condition names and related diseases."""
    conditions = trial.get("protocolSection", {}).get("conditionsModule", {}).get("conditions", [])
    browse_terms = trial.get("protocolSection", {}).get("derivedSection", {}).get("conditionBrowseModule", {}).get("browseLeaves", [])

    all_terms = set(' '.join(conditions).split(' '))
    all_terms.update([term["name"] for term in browse_terms])

    user_words = set(user_response.lower().split())
    match_count = sum(1 for term in all_terms for word in user_words if fuzz.ratio(term.lower(), word) >= 80)

    return match_count >= 1  # Ensure at least 1 word matches for accuracy


def check_age_eligibility(trial, user_response):
    """Checks if user age matches the trial eligibility criteria."""
    if user_response.isdigit():
        user_age = int(user_response)
        min_age = trial.get("protocolSection", {}).get("eligibilityModule", {}).get("minimumAge", "0 Years").split()[0]
        max_age = trial.get("protocolSection", {}).get("eligibilityModule", {}).get("maximumAge", "999 Years").split()[0]

        min_age = int(min_age) if min_age.isdigit() else 0
        max_age = int(max_age) if max_age.isdigit() else 999

        return min_age <= user_age <= max_age

    return False


def check_gender_eligibility(trial, user_response):
    """Checks if user gender matches the trial eligibility criteria."""
    gender = trial.get("protocolSection", {}).get("eligibilityModule", {}).get("gender", "").lower()
    return user_response.lower() in gender


def check_health_status_eligibility(trial, user_response):
    """Checks if user health status matches the trial eligibility criteria."""
    health_status = trial.get("protocolSection", {}).get("eligibilityModule", {}).get("healthStatus", "").lower()
    return user_response.lower() in health_status


def check_location(trial, user_response):
    """Checks if user location matches the trial's location."""
    locations = trial.get("protocolSection", {}).get("contactsLocationsModule", {}).get("locations", [])

    for loc in locations:
        if any(user_response.lower() in loc[field].lower() for field in ["city", "state", "country"] if field in loc):
            return True
    return False


def check_recruitment_status(trial, user_response, threshold=80):
    """Filters trials based on recruitment status using fuzzy matching."""
    recruitment_status = trial.get("protocolSection", {}).get("statusModule", {}).get("overallStatus", "").lower()
    user_response = user_response.lower()

    statuses = ["recruiting", "enrolling", "completed"]

    for status in statuses:
        if fuzz.partial_ratio(user_response, status) >= threshold and fuzz.partial_ratio(recruitment_status, status) >= threshold:
            return True

    return False


def check_intervention(trial, user_response):
    """Filters trials based on intervention type (e.g., gene therapy, drug, biological)."""
    interventions = trial.get("protocolSection", {}).get("armsInterventionsModule", {}).get("interventions", [])

    for intervention in interventions:
        intervention_type = intervention.get("type", "").lower()
        intervention_name = intervention.get("name", "").lower()

        if user_response.lower() in intervention_type or user_response.lower() in intervention_name:
            return True
    return False


def check_study_type(trial, user_response):
    """Filters trials based on study type (e.g., observational, interventional)."""
    study_type = trial.get("protocolSection", {}).get("designModule", {}).get("studyType", "").lower()

    if user_response.lower() in study_type:
        return True
    return False


def check_sponsor(trial, user_response):
    """Filters trials based on sponsor or organization name."""
    sponsor = trial.get("protocolSection", {}).get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name", "").lower()

    if user_response.lower() in sponsor:
        return True
    return False


def check_expanded_access(trial, user_response):
    """Filters trials based on whether they offer expanded access."""
    has_expanded_access = trial.get("protocolSection", {}).get("statusModule", {}).get("expandedAccessInfo", {}).get("hasExpandedAccess", False)

    return "expanded access" in user_response.lower() and has_expanded_access


def check_genetic_study(trial, user_response):
    """Filters trials that involve genetic studies or DNA samples."""
    bio_spec = trial.get("protocolSection", {}).get("designModule", {}).get("bioSpec", {}).get("retention", "").lower()

    return "dna" in user_response.lower() or "genetic" in bio_spec


def check_sample_size(trial, user_response):
    """Filters trials based on the estimated number of enrolled participants."""
    if user_response.isdigit():
        user_sample_size = int(user_response)
        trial_enrollment = trial.get("protocolSection", {}).get("designModule", {}).get("enrollmentInfo", {}).get("count", 0)

        return user_sample_size <= trial_enrollment

    return False


def check_fda_regulation(trial, user_response):
    """Filters trials that are regulated by the FDA."""
    is_fda_regulated = trial.get("protocolSection", {}).get("oversightModule", {}).get("isFdaRegulatedDrug", False)

    return "fda regulated" in user_response.lower() and is_fda_regulated


def generate_sub_questions(trials, previous_responses):
    """Generates sub-questions for each chunk of trials."""
    # sub_questions = []
    #
    # for chunk in trials:
    #     chunk = str(chunk)[0:4081]
    context = f"""
    Given the following clinical trials: {trials}

    The user has provided these responses so far: {previous_responses}

    Generate a relevant question to refine the selection further.
    """

    response = co.generate(
        model="command",
        prompt=context,
        max_tokens=50,
        truncate="START"
    )

    return  response.generations[0].text.strip()



if __name__ == "__main__":
    app.run(debug=True)


