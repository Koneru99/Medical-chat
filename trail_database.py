import json
from typing import List, Optional
from dataclasses import dataclass, field
import re
from thefuzz import fuzz

@dataclass
class GeoPoint:
    lat: Optional[float] = None
    lon: Optional[float] = None

@dataclass
class Contact:
    name: Optional[str] = None
    role: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None

@dataclass
class Location:
    facility: Optional[str] = None
    status: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None
    country: Optional[str] = None
    contacts: List[Contact] = field(default_factory=list)
    geoPoint: Optional[GeoPoint] = None

@dataclass
class ContactsLocationsModule:
    locations: List[Location] = field(default_factory=list)

@dataclass
class ConditionsModule:
    conditions: List[str] = field(default_factory=list)

@dataclass
class IdentificationModule:
    nctId: Optional[str] = None
    briefTitle: Optional[str] = None

@dataclass
class EligibilityModule:
    minimumAge: Optional[str] = None
    maximumAge: Optional[str] = None
    

@dataclass
class ProtocolSection:
    identificationModule: IdentificationModule
    conditionsModule: ConditionsModule
    eligibilityModule: EligibilityModule
    contactsLocationsModule: ContactsLocationsModule

@dataclass
class StudyData:
    protocolSection: ProtocolSection

# Function to convert dictionaries to dataclass instances, handling missing fields
def dict_to_dataclass(data: dict) -> StudyData:
    protocol_section = data.get("protocolSection", {})

    return StudyData(
        protocolSection=ProtocolSection(
            identificationModule=IdentificationModule(
                nctId=protocol_section.get("identificationModule", {}).get("nctId"),
                briefTitle=protocol_section.get("identificationModule", {}).get("briefTitle")
            ),
            conditionsModule=ConditionsModule(
                conditions=protocol_section.get("conditionsModule", {}).get("conditions", [])
            ),
            eligibilityModule=EligibilityModule(
                minimumAge=protocol_section.get("eligibilityModule", {}).get("minimumAge"),
                maximumAge=protocol_section.get("eligibilityModule", {}).get("maximumAge")
                ),
            contactsLocationsModule=ContactsLocationsModule(
                locations=[
                    Location(
                        facility=loc.get("facility"),
                        status=loc.get("status"),
                        city=loc.get("city"),
                        state=loc.get("state"),
                        zip=loc.get("zip"),
                        country=loc.get("country"),
                        contacts=[
                            Contact(
                                name=contact.get("name"),
                                role=contact.get("role"),
                                phone=contact.get("phone"),
                                email=contact.get("email")
                            ) for contact in loc.get("contacts", [])
                        ],
                        geoPoint=GeoPoint(
                            lat=loc.get("geoPoint", {}).get("lat"),
                            lon=loc.get("geoPoint", {}).get("lon")
                        ) if "geoPoint" in loc else None
                    ) for loc in protocol_section.get("contactsLocationsModule", {}).get("locations", [])
                ]
            )
        )
    )

# Load JSON array and parse it
def load_study_data_from_json(json_data: str) -> List[StudyData]:
    data_list = json.loads(json_data)  # Parse the JSON array string
    return [dict_to_dataclass(data) for data in data_list]


def read_data():
    with open("studies.json", "r",encoding='latin-1') as file:
        json_data = file.read()
    return load_study_data_from_json(json_data)
 
def getTheCountBasedOnConditions(conditionData,disease):

    listitems = []
    disease_words = set(disease.lower().split())  # Convert disease to lowercase and split into words

    for i in conditionData:
        conditions = i.protocolSection.conditionsModule.conditions
        for j in conditions:
            condition_words = set(j.lower().split())  # Convert condition to lowercase and split into words

            # Check for a fuzzy match (threshold 80%)
            if any(fuzz.ratio(word1, word2) >= 80 for word1 in disease_words for word2 in condition_words):
                print(disease, j)
                listitems.append(i)
                break  # Exit inner loop once a match is found

    return listitems


def extractTrailsBasedOnlocations(conditionData,location):
    filteredAddress = []
    threshold = 80  # Adjust threshold for strictness

    for i in conditionData:
        locations = i.protocolSection.contactsLocationsModule.locations
        filtered_locations_address = [
            loc for loc in locations
            if loc.city is not None and fuzz.ratio(location.lower(), loc.city.lower()) >= threshold
               or loc.state is not None and fuzz.ratio(location.lower(), loc.state.lower()) >= threshold
               or loc.country is not None and fuzz.ratio(location.lower(), loc.country.lower()) >= threshold
        ]

        if filtered_locations_address:
            filteredAddress.append(filtered_locations_address[0])

    return filteredAddress


def getTrialsBasedOnAge(conditionData,age):
    listitems=[]
    for i in conditionData:
        ageConditions = i.protocolSection.eligibilityModule
        
        if ageConditions.minimumAge is not None:
            mini = int(re.search(r'\d+', ageConditions.minimumAge).group())
        else:
            mini = 0 
        if ageConditions.maximumAge is not None:
            maxi = int(re.search(r'\d+', ageConditions.maximumAge).group())
        else:
            maxi = 100  
        
        if age>=mini and age<=maxi:
            listitems.append(i)
    return listitems

