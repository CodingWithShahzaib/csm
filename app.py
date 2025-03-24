import os
import torch
import torchaudio
from generator import load_csm_1b

# Disable Triton compilation
os.environ["NO_TORCH_COMPILE"] = "1"

def main():
    # Select the best available device, skipping MPS due to float64 limitations
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    generator = load_csm_1b(device=device)
    print("Model loaded successfully!")
    print(f"Sample rate: {generator.sample_rate}")
    
    # Generate a short audio sample
    print("Generating audio... (this may take a while on CPU)")
    text = "osama is a good boy"
    audio = generator.generate(
        text=text,
        speaker=99,
        context=[],
        max_audio_length_ms=3000,  # Short sample (3 seconds max)
        temperature=0.9,
        topk=50,
    )
    
    # Save the generated audio
    output_file = "osama_is_a_good_boy.wav"
    torchaudio.save(
        output_file,
        audio.unsqueeze(0).cpu(),
        generator.sample_rate
    )
    print(f"Successfully generated audio saved to {output_file}")

if __name__ == "__main__":
    main()