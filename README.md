# 🎈 Play-with-Transformer-based-Models

**[Try the app 👀](https://transformer-based-models-playground.streamlit.app/)**

This is a web-based application built with **Streamlit** that allows users to interact with various transformer-based models for a wide range of machine learning tasks. The app integrates Hugging Face's **Transformers** library and other popular Python tools, enabling real-time processing of text and images. 

Users can leverage pre-trained models or upload their own custom models for tasks like **text generation**, **sentiment analysis**, **summarization**, **zero-shot classification**, and **image classification**. The app also includes interactive features and data visualizations to make model outputs more intuitive and engaging.

### **Key Features**
- **Pre-trained Models**: Access a wide range of models for tasks such as summarization, sentiment analysis, and more.
- **Custom Model Support**: Upload and use your own PyTorch model files (`.bin`, `.pt`, `.zip`) for any supported task.
- **Real-time Processing**: Perform tasks like text summarization, sentiment analysis, and image classification instantly.
- **Interactive Interface**: Easily select tasks, models, and input data through a simple, user-friendly web interface.
- **Data Visualizations**: Visual outputs such as word clouds, probability distributions, and heatmaps to enhance understanding of model predictions.

### **Available Tasks**
1. **Summarization**: Input long text and generate a concise summary using powerful models.
2. **Sentiment Analysis**: Analyze and detect the sentiment (positive, negative, neutral) of the input text.
3. **Zero-Shot Classification**: Classify text into predefined categories without additional model training.
4. **Text Generation**: Generate text based on an initial prompt using methods like sampling or beam search.
5. **Image Classification**: Upload images for classification using pre-trained models or custom labels.

### **Custom Model Support**
- Upload your own custom models for any of the tasks. The app automatically loads and uses the models for inference without additional setup.

### **Data Visualizations**
- **Word Clouds**: For sentiment analysis tasks, visualize the most frequent words in the text.
- **Probability Distributions**: For text generation tasks, view the probability distribution of the next predicted word in a bar chart.
- **Heatmaps**: For image classification tasks, visualize the model's attention on different parts of the image.

# 📌 How It Works

1. **Select a Task**: Choose from tasks like summarization, sentiment analysis, zero-shot classification, text generation, or image classification.
2. **Model Selection**: Pick from a list of pre-trained models or upload your custom model.
3. **Input Data**: Provide the necessary input for the selected task (e.g., text for sentiment analysis or summarization, image for classification).
4. **Get Results**: The model processes the input and returns results, including text, classifications, or visual outputs (e.g., word clouds, bar charts).
5. **Visual Outputs**: Depending on the task, graphical outputs like word clouds, heatmaps, or probability distributions are generated and displayed.

### **Example Usage**

- **Summarization**:
  - **Task**: Summarize a given piece of text.
  - **Input**: Paste a large block of text.
  - **Output**: A concise summary of the text.
  - [View the Summarization results](results/summarization_results.pdf)
  
- **Sentiment Analysis**:
  - **Task**: Analyze the sentiment of a text.
  - **Input**: Paste a review or social media post.
  - **Output**: Sentiment result (positive/negative) with a confidence score, plus a word cloud of frequently occurring terms.
  - [View the Sentiment Analysis results](results/sentiment-analysis_results.pdf)
   
- **Zero-Shot Classification**:
  - **Task**: Classify text into predefined categories.
  - **Input**: Paste text and provide candidate labels (e.g., 'Technology', 'Health').
  - **Output**: Classification result with the most likely category and its confidence score.
  - [View the Zero-Shot Classification results](results/zero-shot-classification_results.pdf)
   
- **Text Generation**:
  - **Task**: Generate text based on an input prompt.
  - **Input**: Provide an initial prompt.
  - **Output**: Continuation of the prompt with generated text, and a chart showing the probability distribution of the next word.
  - [View the Text Generation results](results/text-generation_results.pdf)
    
- **Image Classification**:
  - **Task**: Classify an image.
  - **Input**: Upload an image.
  - **Output**: Classification results with confidence scores and optional custom labels.
  - [View the Image Classification results](results/image-classification_results.pdf)
    
### **Getting Started**
Simply select your desired task, choose from a variety of pre-trained models or upload your custom model, and interact with the outputs in real time. With its interactive features and intuitive interface, this app is perfect for both beginners and advanced users exploring the power of transformer-based models.

---

### **Run It Locally**

To run this app locally, follow these steps:

#### 1. Clone the repository
#### 2. install neccessary libraries mentioned in (requirements.txt)
#### 3. type "streamlit run compare_models_and_task.py" in your terminal.

### **Thank You!!!**


