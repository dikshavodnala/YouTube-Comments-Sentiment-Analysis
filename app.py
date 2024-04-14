import os
import googleapiclient.discovery
from flask import request, Flask, render_template
import pandas as pd
import re
import emoji
import string
from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
import re
import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
nltk.download('vader_lexicon')
import base64
from io import BytesIO

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # if request.method=="POST":

    link = request.form["url_name"]
    vid_id = link[-11:]

    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = "AIzaSyCCXaC4Oo28brpUYOwVSStDEZFTahDnBHc"

    # DEVELOPER_KEY = "ENTER_YOUR_DEVELOPER_KEY"

    youtube = googleapiclient.discovery.build(api_service_name,
                            api_version, developerKey=DEVELOPER_KEY)
    # Make a request to youtube API
    link_req = youtube.commentThreads().list(
        part='snippet',
        videoId=vid_id,
        maxResults=100
    )

    # Get a response from API
    response = link_req.execute()

    video_snippet = response['items'][0]['snippet']
    uploader_channel_id = video_snippet['channelId']

    # authorname = []
    comments = []

    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        if comment['authorChannelId']['value'] != uploader_channel_id:
            comments.append(comment['textOriginal'])

    df = pd.DataFrame(comments, columns=['original comments'])

    ## Data Cleaning

    for i in range(len(df)):
        comment = df.loc[i, 'original comments']

        # Convert comments to lowercase
        df.loc[i, 'original comments'] = " ".join(x.lower() for x in comment.split())

        # Remove emojis
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F700-\U0001F77F"  # alchemical symbols
                                   u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                                   u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                                   u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                                   u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                                   u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                                   u"\U00002702-\U000027B0"  # Dingbats
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        df.loc[i, 'original comments'] = emoji_pattern.sub('', df.loc[i, 'original comments'])

        # Remove punctuation marks
        df.loc[i, 'original comments'] = "".join(
            [char for char in df.loc[i, 'original comments'] if char not in string.punctuation])

        # Remove non-alphabetic characters and digits
        df.loc[i, 'original comments'] = re.sub('[^a-zA-Z]', ' ', df.loc[i, 'original comments'])

        # Remove multiple whitespaces to single space
        df.loc[i, 'original comments'] = re.sub(r'\s+', ' ', df.loc[i, 'original comments']).strip()

        # Remove stopwords
        df.loc[i, 'original comments'] = " ".join([word for word in df.loc[i, 'original comments'].split() if
                                                   word.lower() not in set(stopwords.words('english'))])

    # Remove rows with empty comments
    df = df[df['original comments'].str.len() > 0]
    df.reset_index(drop=True, inplace=True)

    positive_comments = []
    negative_comments = []
    neutral_comments = []

    # Function to get overall sentiment using mean
    def get_overall_sentiment(df):
        # Initialize the SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()

        # Function to get sentiment score
        def get_sentiment_score(text):
            return sia.polarity_scores(text)['compound']

        # Apply the function to each comment in the DataFrame
        df['sentiment_score'] = df['original comments'].apply(get_sentiment_score)

        # Calculate mean sentiment score
        mean_sentiment_score = df['sentiment_score'].mean()
        # print(mean_sentiment_score)

        # Append comments to respective lists
        for index, row in df.iterrows():
            comment = row['original comments']
            score = get_sentiment_score(comment)

            if score > 0.05:
                positive_comments.append(comment)
            elif score < -0.05:
                negative_comments.append(comment)
            else:
                neutral_comments.append(comment)

        # Determine overall sentiment label
        if mean_sentiment_score > 0.05:
            overall_sentiment = 'positive'
        elif mean_sentiment_score < -0.05:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'

        return overall_sentiment

    # Get overall sentiment
    overall_sentiment = get_overall_sentiment(df)
    mean_sentiment_score = df['sentiment_score'].mean()

    # Display the results

    print("Overall Sentiment: {}".format(overall_sentiment))
    print("Average Sentiment Score:",mean_sentiment_score)

    # return render_template("index.html",prediction_text=f"The Overall Sentiment is:{overall_sentiment} and the Average Sentiment Score is {mean_sentiment_score} ")
    positive_count = len(positive_comments)
    negative_count = len(negative_comments)
    neutral_count = len(neutral_comments)

    labels = ['Positive', 'Negative', 'Neutral']
    comment_counts = [positive_count, negative_count, neutral_count]

    # Plot bar chart
    plt.bar(labels, comment_counts, color=['blue', 'red', 'grey'])
    plt.xlabel('Sentiment')
    plt.ylabel('Comment Count')
    plt.title('Sentiment Analysis of Comments')

    # Save the plot to a BytesIO object
    bar_chart_image = BytesIO()
    plt.savefig(bar_chart_image, format='png')
    bar_chart_image.seek(0)
    plt.close()

    # Word cloud
    all_words = ' '.join([i for i in df["original comments"]])
    wordcloud = WordCloud(width=600, max_font_size=50, height=350, max_words=100, background_color="white").generate(
        all_words)

    # Plot word cloud
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')

    # Save the plot to a BytesIO object
    wordcloud_image = BytesIO()
    plt.savefig(wordcloud_image, format='png')
    wordcloud_image.seek(0)
    plt.close()

    # Convert BytesIO objects to base64-encoded strings
    bar_chart_base64 = base64.b64encode(bar_chart_image.read()).decode('utf-8')
    wordcloud_base64 = base64.b64encode(wordcloud_image.read()).decode('utf-8')

    return render_template("index.html",
                           prediction_text=f"The Overall Sentiment is:{overall_sentiment} and the Average Sentiment Score is {mean_sentiment_score}",
                           bar_chart_image=bar_chart_base64,
                           wordcloud_image=wordcloud_base64)


if __name__=="__main__":
    app.run(debug=True,threaded=False)


