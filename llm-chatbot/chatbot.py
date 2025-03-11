from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import gradio as gr

# Use Facebook's BlenderBot model (distilled version, which is lighter)
model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

def generate_response(user_input):
    # Tokenize the input text
    inputs = tokenizer([user_input], return_tensors="pt")
    
    # Generate a response with controlled decoding
    output_ids = model.generate(
        inputs["input_ids"],
        max_length=150,            # Maximum length of the generated sequence
        do_sample=True,            # Enable sampling for variety
        temperature=0.7,           # Lower temperature for more focused responses
        top_p=0.95,                # Nucleus sampling: consider tokens with 95% cumulative probability
        pad_token_id=tokenizer.eos_token_id  # Use the end-of-sequence token for padding
    )
    
    # Decode the generated tokens to text, skipping special tokens
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

# Set up a Gradio interface for easy testing
iface = gr.Interface(
    fn=generate_response,
    inputs="text",
    outputs="text",
    title="BlenderBot Chatbot",
    description="Chat with Facebook's BlenderBot (400M-distill) model."
)

if __name__ == "__main__":
    iface.launch()

