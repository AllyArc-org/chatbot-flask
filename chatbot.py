import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify, render_template

# Load the pre-trained model and tokenizer
model_name = "AllyArc/llama_allyarc"  # Replace with your desired model name
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize Flask app
app = Flask(__name__)

# Initialize conversation history
conversation_history = ""

# Chatbot logic
def generate_response(input_text, conversation_history):
    print("Generating response...")
    print("Input Text:", input_text)
    print("Conversation History:", conversation_history)

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

    print("Generated Response:", response)

    # Extract only the first instance of the assistant's response
    assistant_response = response.split("User:", 1)[0].strip()

    print("Assistant Response:", assistant_response)

    return assistant_response

# Response filtering function
def filter_response(response, user_input):
    print("Filtering response...")
    print("Response:", response)
    print("User Input:", user_input)

    # Implement your response filtering logic here
    # Example: Check for inappropriate or nonsensical content
    if "inappropriate" in response.lower():
        response = "I apologize, but I cannot provide an appropriate response."
    
    print("Filtered Response:", response)
    
    return response

# Serve the HTML file
@app.route('/')
def home():
    return render_template('chatbot.html')

# API endpoint for chatbot
@app.route('/chatbot', methods=['POST'])
def chatbot():
    global conversation_history

    # Get the user input from the request
    user_input = request.json.get('input', '')

    print("User Input:", user_input)

    # Generate the assistant's response based on the user input and conversation history
    assistant_response = generate_response(user_input, conversation_history)

    # Filter the generated response (if necessary)
    assistant_response = filter_response(assistant_response, user_input)

    # Update the conversation history
    conversation_history += "User: " + user_input + "\nAssistant: " + assistant_response + "\n"

    print("Updated Conversation History:", conversation_history)

    # Return the response as JSON
    return jsonify({'response': assistant_response})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)