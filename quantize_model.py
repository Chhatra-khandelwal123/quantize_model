from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_quantized_model_4bit(model_name: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",          # Normal Float 4 bits, best accuracy
        bnb_4bit_compute_dtype="bfloat16",  # Computation precision
        low_cpu_mem_usage=True               # Reduce CPU RAM usage
    )

    print(f"Loading {model_name} in 4-bit quantized mode...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"  # Automatically put layers on GPU/CPU as available
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer

def main():
    model_name = "meta-llama/Llama-2-7b-hf"  # Change to your model
    model, tokenizer = load_quantized_model_4bit(model_name)

    prompt = "Explain quantum computing in simple terms."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate text (adjust max_length and other parameters as needed)
    outputs = model.generate(inputs.input_ids, max_length=100)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated Text:\n{text}")

if __name__ == "__main__":
    main()
