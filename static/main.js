let map;
let isMap=true
function sendMessage() {
    const userInput = document.getElementById("user-input").value;
    if (userInput.trim() === "") return;

    // Append user's message to chat box
    const chatBox = document.getElementById("chat-box");
    chatBox.innerHTML += `<div class="message user-message">You: ${userInput}</div>`;

    // Scroll chatbox down as messages come in
    chatBox.scrollTop = chatBox.scrollHeight;

    // Send user message to the Flask server
    fetch('/send_message', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userInput }),
    })
    .then(response => response.json())
    .then(data => {
        // Append server's response to the chat box
        chatBox.innerHTML += `<div class="message bot-message">Bot: ${data.response}</div>`;
        
        // Check if geoPoints are included in the response
        if (data.geoPoints && data.geoPoints.length) {
            // Append the map directly to the chat box
                chatBox.innerHTML += `<div id="map" style="display: block; width: 100%; height: 400px; margin-top: 20px;"></div>`;
                isMap=false;


            // Initialize the map with geoPoints
            setTimeout(() => {
                console.log(document.getElementById("map"));
                document.getElementById("map").style.display = "block";
                initMap(data.geoPoints);
            }, 100);
        } else {
            console.log("No geo points received from the backend.");
        }
        
        if (data.facilities && data.facilities.length > 0) {
            data.facilities.forEach(facility => {
                let facilityInfo = `<div class="facility-info" style="margin-top: 20px;">`;
        
                if (facility.facility) {
                    facilityInfo += `<h4>${facility.facility}</h4>`;
                }
        
                if (facility.status) {
                    facilityInfo += `<p>Status: ${facility.status}</p>`;
                }
        
                if (facility.city || facility.state || facility.zip || facility.country) {
                    facilityInfo += `<p>Location: ${facility.city || ''}, ${facility.state || ''} ${facility.zip || ''}, ${facility.country || ''}</p>`;
                }
        
                if (facility.contacts && facility.contacts.length > 0) {
                    const contact = facility.contacts[0]; // Assuming you want only the first contact
                    facilityInfo += `<h5>Contact Information:</h5>`;
        
                    if (contact.name) {
                        facilityInfo += `<p>Name: ${contact.name}</p>`;
                    }
        
                    if (contact.role) {
                        facilityInfo += `<p>Role: ${contact.role}</p>`;
                    }
        
                    if (contact.phone) {
                        facilityInfo += `<p>Phone: ${contact.phone}</p>`;
                    }
        
                    if (contact.email) {
                        facilityInfo += `<p>Email: <a href="mailto:${contact.email}">${contact.email}</a></p>`;
                    }
                }
        
                facilityInfo += `</div>`;
                isMap=true
                chatBox.innerHTML += facilityInfo;
            });
        }
        
        // Scroll down as new messages come in
        chatBox.scrollTop = chatBox.scrollHeight;
    })
    .catch(error => {
        console.error('Error sending message:', error);
    });

    

    // Clear input field
    document.getElementById("user-input").value = "";
}

function initMap(geoPoints) {
    // Create the map centered on the first geo-point
    map = new google.maps.Map(document.getElementById("map"), {
        zoom: 8,
        center: { lat: geoPoints[0][0], lng: geoPoints[0][1] },
    });

    // Add markers for each geo-point
    geoPoints.forEach(point => {
        new google.maps.Marker({
            position: { lat: point[0], lng: point[1] },
            map: map,
        });
    });

    // Optionally adjust the map's bounds to fit all markers
    adjustMapBounds(geoPoints);
}

function adjustMapBounds(geoPoints) {
    const bounds = new google.maps.LatLngBounds();
    geoPoints.forEach(point => bounds.extend(new google.maps.LatLng(point[0], point[1])));
    map.fitBounds(bounds);
}
