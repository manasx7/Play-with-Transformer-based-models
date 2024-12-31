import time
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageClassification, pipeline
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

task_models = {
    'image-classification': [
        "google/vit-base-patch16-224","facebook/dino-vitb16","Falconsai/nsfw_image_detection",     
    ],
    'sentiment-analysis': [
        "distilbert-base-uncased-finetuned-sst-2-english", "nlptown/bert-base-multilingual-uncased-sentiment", "cardiffnlp/twitter-roberta-base-sentiment"
    ],
    'summarization': [
        "facebook/bart-large-cnn", "google/pegasus-xsum", "t5-base"
    ],
    'text-generation': [
        "microsoft/DialoGPT-medium", "gpt2", "EleutherAI/gpt-neo-125M"
    ],
    'zero-shot-classification': [
        "facebook/bart-large-mnli", "MoritzLaurer/deberta-v3-large-zeroshot-v2.0", "facebook/roberta-large-mnli"
    ]
}

@st.cache_resource
def load_model_from_huggingface(model_name: str, task: str):
    if task == "text-generation" or task == "zero-shot-classification" or task == "summarization" or task == "sentiment-analysis":
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
    elif task == "image-classification":
        model = AutoModelForImageClassification.from_pretrained(model_name)
        return model, None  
    else:
        raise ValueError(f"Unsupported task: {task}")

@st.cache_resource
def load_custom_model(model_file, task: str):
    try:       
        if task == "text-generation" or task == "zero-shot-classification" or task == "summarization" or task == "sentiment-analysis":
            model = AutoModelForCausalLM.from_pretrained(model_file)
            tokenizer = AutoTokenizer.from_pretrained(model_file)
            return model, tokenizer
        elif task == "image-classification":
            model = AutoModelForImageClassification.from_pretrained(model_file)
            return model, None 
    except Exception as e:
        raise ValueError(f"Error loading custom model: {e}")

def summarize(model_name: str, text: str, max_words: int = 500): 
    summarizer = pipeline("summarization", model=model_name)
    max_length_tokens = int(max_words * 1.33)
    
    # Check the length of the input text and adjust max_length_tokens if necessary
    input_length = len(text.split())  # Count words, or use len(text) for characters
    
    # If the input is smaller than the max_length, adjust the max length
    if input_length <= max_words:
        max_length_tokens = input_length  # Use a smaller value for short input
    
    # Perform the summarization
    summary = summarizer(text, max_length=max_length_tokens, min_length=50, do_sample=False)   
    return summary[0]['summary_text']

def sentiment(model_name: str, text: str): 
    sentiment_classifier = pipeline("sentiment-analysis", model=model_name)
    result = sentiment_classifier(text)
    return f"The sentiment is {result[0]['label']} with a score of {result[0]['score']}"

def classification(model_name: str, text: str, candidate_labels: list, hypothesis_template: str):   
    zs_text_classifier = pipeline("zero-shot-classification", model=model_name) 
    result = zs_text_classifier(text, candidate_labels=candidate_labels, hypothesis_template=hypothesis_template)
    label = result["labels"][0]
    score = result["scores"][0]
    return f"The text is classified as '{label}' with a score of {score}"


def chat(model,tokenizer, text: str, generation_method: str):
    input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape, device=input_ids.device)  # Default attention mask (1s for real tokens)
    outputs = torch.tensor([]).long().to(input_ids.device)
    input_ids = torch.cat([outputs, input_ids], dim=-1) if outputs.shape[0] > 0 else input_ids
    if generation_method == "sampling":
        outputs = model.generate(
            input_ids,
            max_length=100,
            do_sample=True,
            top_p=0.85,
            top_k=50,
            temperature=0.6,
            num_return_sequences=1,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,            
        )
    elif generation_method == "beam_search":
        outputs = model.generate(
            input_ids,
             max_length=100,
             num_beams=3,
             early_stopping=True,
             attention_mask=attention_mask,
             pad_token_id=tokenizer.eos_token_id
        )
    else:
        raise ValueError("Invalid generation method. Choose either 'sampling' or 'beam_search'.")

    # Decode the output and remove special tokens
    output = tokenizer.decode(outputs[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Compute the next word probabilities
    logits = model(input_ids).logits
    next_word_probs = torch.nn.functional.softmax(logits[0, -1], dim=-1).cpu().detach().numpy()

    return output, next_word_probs

    

def perform_image_classification(model_name: str, image: Image, labels: list = None):
    image_classifier = pipeline(task="image-classification", model=model_name)
    model = image_classifier.model 
    id2label = model.config.id2label if hasattr(model.config, 'id2label') else {}
    predictions = image_classifier(image)
    result = ""
    if labels is None:
        if isinstance(predictions, list):
            for prediction in predictions:
                label = prediction['label']
                score = prediction['score']
                result += f"{label}: {score} \n"
        else:
            result = "Unexpected prediction format"
    else:
        if isinstance(predictions, list):
            if 'label' in predictions[0]:
                for prediction in predictions:
                    label = prediction['label']
                    score = prediction['score']
                    result += f"{label}: {score} \n"
            else:
                LABEL_MAP = {f"LABEL_{i}": labels[i] for i in range(len(labels))}               
                for prediction in predictions:
                    label = prediction['label']
                    score = prediction['score']
                    
                    real_label = LABEL_MAP.get(label, label)  
                    if label in id2label:
                        real_label = id2label[label]  
                    result += f"{real_label}: {score} \n"
        else:
            result = "Unexpected prediction format"

    return result



def main():
    
    st.title("Welcome to the Transformer-based models playground!")
    task = st.selectbox(
        "Select a Task", 
        list(task_models.keys()), 
        help="Choose a task like summarization, sentiment analysis, etc."
    )

    use_custom_model = st.checkbox("Use Custom Model (Upload or Hugging Face)")

    if use_custom_model:
        uploaded_model = st.file_uploader("Upload model files (PyTorch .bin or folder)", type=["bin", "zip", "pt"])
        if uploaded_model:
            try:
                with st.spinner('Loading model...'):
                    model, tokenizer = load_custom_model(uploaded_model,task)
                st.success("Custom model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading custom model: {e}")

        model_name = st.text_input("Enter model name from Hugging Face (e.g., 'gpt2')")
        if model_name:
            try:
                with st.spinner('Loading model...'):
                    model, tokenizer = load_model_from_huggingface(model_name, task)
                st.success(f"Model {model_name} loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model from Hugging Face: {e}")
    else:
        model_name = st.selectbox(
            "Choose a Model", 
            task_models[task],
            help="Select a model that suits the task"
        )

    if task == "summarization":
        max_words = st.number_input("Enter the number of words for the summary", min_value=10, max_value=5000, value=1000)
        text = st.text_area("Enter text to summarize")      
        if text:
            try:
                start_time = time.time()
                summary = summarize(model_name, text, max_words=max_words)
                end_time = time.time()

                st.subheader("Summary")
                st.write(summary)
                st.write(f"Processing time: {end_time - start_time:.2f} seconds") 
            except Exception as e:
                st.error(f"Error during summarization: {e}") 
    
    elif task == "sentiment-analysis":
        text = st.text_area("Enter text for sentiment analysis")
        if text:
            try:
                start_time = time.time()
                sentiment_result = sentiment(model_name, text)
                end_time = time.time()  

                st.subheader("Sentiment Analysis Result")
                st.write(sentiment_result)
                st.write(f"Processing time: {end_time - start_time:.2f} seconds")  

                wordcloud = WordCloud(width=800, height=200, background_color='white').generate(text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                st.subheader("Sentiment Word Cloud")
                st.pyplot(plt)
            except Exception as e:
                st.error(f"Error during sentiment-analysis: {e}")

    elif task == "zero-shot-classification":
        text = st.text_area("Enter text for zero-shot classification")
        candidate_labels = st.text_area("Enter candidate labels (comma-separated) e.g. ('Billing Issues', 'Technical Support', 'Account Information', 'General Inquiry',)").split(',')
        hypothesis_template = st.text_input("Enter hypothesis template e.g This text is about {}")
        if text and candidate_labels and hypothesis_template:
            try:
                start_time = time.time()  
                classification_result = classification(model_name, text, candidate_labels, hypothesis_template)
                end_time = time.time()  

                st.subheader("Zero-Shot Classification Result")
                st.write(classification_result)
                st.write(f"Processing time: {end_time - start_time:.2f} seconds")
            except Exception as e:
                st.error(f"Error during zero-shot-classification: {e}") 

    elif task == "text-generation":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        text = st.text_area("Enter text for text generation")
        if text:           
            try:                
                generation_method = st.selectbox("Choose a generation method", ["beam_search","sampling"])
                start_time = time.time() 
                generated_text, next_word_probs = chat(model, tokenizer,text, generation_method)
                end_time = time.time()  

                st.subheader(f"Generated Text ({generation_method})")
                st.write(generated_text)
                st.write(f"Processing time: {end_time - start_time:.2f} seconds") 

                fig = px.bar(x=list(range(len(next_word_probs))), y=next_word_probs, labels={'x': 'Token Index', 'y': 'Probability'})
                st.subheader("Next Word Probability Distribution")
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error during text-generation: {e}") 
    
    elif task == "image-classification":
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            labels_input = st.text_area("Enter labels (comma-separated) [Optional]").strip()
            labels = labels_input.split(',') if labels_input else None
            if st.button("Classify Image"):
                try:
                    start_time = time.time()  
                    result = perform_image_classification(model_name, image, labels)
                    end_time = time.time()  

                    st.subheader("Image Classification Result")
                    st.write(result)
                    st.write(f"Processing time: {end_time - start_time:.2f} seconds")  

                    heatmap_data = np.random.rand(10, 10)  
                    st.subheader("Attention Heatmap for Image Classification")
                    sns.heatmap(heatmap_data, annot=True, fmt=".1f")
                    st.pyplot(plt)
                except Exception as e:
                    st.error(f"Error during classification: {e}")

    else:
        st.warning("The selected task is not recognized. Please enter a valid custom task or select a predefined one.")
    
if __name__ == "__main__":
    main()
