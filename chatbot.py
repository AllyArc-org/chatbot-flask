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
def generate_response(input_text, conversation_history):
    # Combine the conversation history and user input
    input_text = conversation_history + "User: " + input_text + "\nAssistant:"

    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate the response
    output = model.generate(
        input_ids,
        max_length=150,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Decode the generated response
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove the Assistant: prefix from the response
    response = response.replace("Assistant:", "").strip()

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

    # Get the conversation history from the request
    conversation_history = request.json.get('conversation_history', '')

    # Generate the response based on the user input and conversation history
    response = generate_response(user_input, conversation_history)

    # Filter the generated response
    response = filter_response(response)

    # Append the user input and generated response to the conversation history
    conversation_history += f"User: {user_input}\nAssistant: {response}\n"

    # Return the updated conversation history and response as JSON
    return jsonify({'response': response, 'conversation_history': conversation_history})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)