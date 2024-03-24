import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify, render_template

# Load the pre-trained model and tokenizer
model_name = "AllyArc/llama_allyarc"  # Replace with your desired model name
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize Flask app
app = Flask(__name__)

# Chatbot logic
def generate_response(input_text):
    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate the response
    output = model.generate(
        input_ids,
        max_length=150,  # Increase the maximum response length
        num_return_sequences=1,
        temperature=0.7,  # Adjust the temperature for response diversity
        top_p=0.9,  # Adjust the top_p for response coherence
    )

    # Decode the generated response
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Implement response filtering
    response = filter_response(response)

    return response

# Response filtering function
def filter_response(response):
    # Implement your response filtering logic here
    # Example: Check for inappropriate or nonsensical content
    if "inappropriate" in response.lower():
        response = "I apologize, but I cannot provide an appropriate response."
    
    return response

# Serve the HTML file
@app.route('/')
def home():
    return render_template('chatbot.html')

# API endpoint for chatbot
@app.route('/chatbot', methods=['POST'])
def chatbot():
    # Get the user input from the request
    user_input = request.json.get('input', '')

    # Provide more context to the chatbot
    context = "User: " + user_input + "\nAssistant:"

    # Generate the response
    response = generate_response(context)

    # Return the response as JSON
    return jsonify({'response': response})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)