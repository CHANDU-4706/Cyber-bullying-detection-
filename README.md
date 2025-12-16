# 💻 Cyber Bullying Detection Application (CBDA)

_**Cyber Bullying Detection Application (CBDA)**_ is a robust **Machine Learning Web Application** designed to detect and flag cyberbullying content in text. Built with Python and Streamlit, it leverages Natural Language Processing (NLP) techniques to analyze user inputs and classify them effectively.

![Image](https://user-images.githubusercontent.com/51766689/226756775-745d6676-1e8e-471b-ab2c-15eb7f99c37c.png)

## ⚙️ How it Works
The application follows a sophisticated pipeline to process text and predict bullying behavior:

1.  **Input Processing**: The user inputs text into the application.
2.  **Text Cleaning**: The system removes noise using **Regular Expressions (Regex)**:
    *   URLs, Usernames (@), and Hashtags (#)
    *   Emojis and Special Characters
    *   "RT" (Retweet) markers and Numbers
3.  **Transformation**: The cleaned text undergoes further NLP processing:
    *   **Lowercasing**: Converting all text to lowercase for consistency.
    *   **Tokenization**: Splitting text into individual words.
    *   **Stopword Removal**: Removing common words (e.g., "the", "is") utilizing the **NLTK** library.
    *   **Stemming**: Reducing words to their root form (e.g., "bullying" -> "bulli") using the **PorterStemmer**.
4.  **Vectorization**: The processed text is converted into numerical data using a **TF-IDF Vectorizer**.
5.  **Prediction**: A pre-trained **Random Forest Classifier** analyzes the vector and predicts if the content is:
    *   🔴 **Cyberbullying**
    *   🟢 **Not Cyberbullying**

## 🛠 Technologies & Tools
*   **Python**: Core programming language.
*   **Streamlit**: Framework for the web interface.
*   **Scikit-Learn**: For the Random Forest model and TF-IDF vectorizer.
*   **NLTK**: For natural language processing tasks (tokenization, stopwords).
*   **Pandas**: For data handling.
*   **Pickle**: For loading the pre-trained model and vectorizer.
*   **HTML5 / CSS3**: For additional styling.

## 🏃‍♂️ How to Run

1. **Navigate to the project directory**
   ```bash
   cd "cyber bully project"
   ```

2. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   streamlit run "1_🏠_CBDA.py"
   ```
   *Note: If you encounter an error saying `streamlit` is not recognized, use this command instead:*
   ```bash
   python -m streamlit run "1_🏠_CBDA.py"
   ```

<br>

![Image](https://user-images.githubusercontent.com/51766689/226749340-ca1d14f0-901a-48ae-a645-074a4d3b3410.png)
