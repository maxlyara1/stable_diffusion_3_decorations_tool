import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from diffusers import StableDiffusionPipeline
import torch
from huggingface_hub import login
import os

# Login to Hugging Face
login(token="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")

# Load the translation model
@st.cache_resource
def load_translation_model():
    tokenizer = AutoTokenizer.from_pretrained("utrobinmv/t5_translate_en_ru_zh_large_1024")
    model = AutoModelForSeq2SeqLM.from_pretrained("utrobinmv/t5_translate_en_ru_zh_large_1024")
    return tokenizer, model

# Load the image generation model
@st.cache_resource
def load_image_generation_model():
    try:
        model = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium", 
            torch_dtype=torch.float16
        ).to("cuda")
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None
    return model

# Translation function
def translate_text(tokenizer, model, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

# Image generation function
def generate_image(model, prompt):
    image = model(prompt).images[0]
    return image

# Streamlit interface
def main():
    st.title("Генерация виртуальных декораций в театре")
    
    st.write("Введите текст на русском языке и нажмите кнопку 'Сгенерировать', чтобы получить перевод и изображение.")

    default_prompt = "Генерация виртуальных декораций в театре"
    user_input = st.text_area("Введите ваш текст:", value=default_prompt)
    
    if st.button("Сгенерировать"):
        with st.spinner('Перевод текста...'):
            tokenizer, translation_model = load_translation_model()
            translated_text = translate_text(tokenizer, translation_model, user_input)
            st.success("Перевод завершен.")
            st.write(f"Переведенный текст: {translated_text}")
        
        with st.spinner('Генерация изображения...'):
            image_generation_model = load_image_generation_model()
            if image_generation_model:
                generated_image = generate_image(image_generation_model, translated_text)
                st.success("Изображение сгенерировано.")
                st.image(generated_image)

if __name__ == "__main__":
    main()
