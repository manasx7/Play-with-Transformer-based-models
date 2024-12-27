# 🎈 Play-with-Transformer-based-models
This project is a web-based application using Streamlit that allows users to interact with various transformer-based models for different machine learning tasks.

The app leverages Hugging Face's Transformers library and other popular Python libraries to enable text and image processing tasks. It supports several pre-trained models and allows users to upload custom models for tasks such as text generation, sentiment analysis, summarization, zero-shot classification, and image classification.

Features
    >>>  Pre-trained models from Hugging Face for multiple tasks (e.g., summarization, sentiment analysis, text generation).
    >>>  Custom model support: Upload your own model files (PyTorch .bin, .pt, or .zip).
    >>>  Real-time text and image processing: Perform tasks like sentiment analysis, summarization, and image classification.
    >>>  Interactive interface: Choose tasks, models, and input data through a user-friendly web interface built with Streamlit.
    >>>  Data visualization: Includes graphical outputs such as word clouds, bar charts, and heatmaps for enhanced understanding of model results.

Available Tasks
    1.  Summarization: Input long text and generate a concise summary using pre-trained models.
    2.  Sentiment Analysis: Analyze the sentiment (positive/negative) of input text.
    3.  Zero-Shot Classification: Classify text into predefined categories without additional training.
    4.  Text Generation: Generate text based on a prompt using various methods (sampling or beam search).
    5.  Image Classification: Upload images for classification with pre-trained models or custom labels.

Custom Models
    You can upload your own model files (PyTorch .bin, .pt, or .zip) for any of the supported tasks. The app will automatically load and use these custom models for inference.

Data Visualizations
    Word Clouds: For sentiment analysis, word clouds are generated based on the input text.
    Probability Distributions: For text generation, a bar chart displays the probability distribution of the next predicted word.
    Heatmaps: For image classification, a heatmap is generated to visualize model attention on the image.
    


