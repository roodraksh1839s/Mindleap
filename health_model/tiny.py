from transformers import AutoTokenizer, AutoModelForCausalLM

# Replace this with the path to your saved model folder
model_path = r"C:\Users\roodr\OneDrive\Desktop\My_learnings\SIH\health_model"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

print("TinyLlama is ready! Type 'quit' to exit.")

while True:
    user_input = input("\nInstruction: ")
    if user_input.lower() in ["quit", "exit"]:
        print("Exiting...")
        break

    # Prepare input
    prompt = f"Instruction: {user_input}\nResponse:"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate response
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=200,   # adjust length as needed
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Only show the generated response, not the repeated prompt
    answer = response.replace(prompt, "").strip()
    print(f"Response: {answer}")
