import os
from flask import Flask, request, render_template, jsonify
import pandas as pd
import logging
from googleapiclient.discovery import build
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

logging.basicConfig(filename='youtube_data.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__,static_folder='static')

api_key = '<API KEY>'

logging.info('Initializing the application.')
print("run on localhost:5000")

@app.route('/', methods=['GET', 'POST'])
def example():
    logging.info('Accessed the / route')
    return render_template('example.html')

@app.route('/stat', methods=['GET', 'POST'])
def data():
    if request.method == 'POST':
        channel_name = request.form['channel_name']  # Get the channel name from the form
        if channel_name:
            logging.info(f'Starting data retrieval for channel: {channel_name}')
            channel_stats = get_channel_stats(channel_name)  # Call your web scraping function
            if channel_stats:
                logging.info('Channel stats retrieved successfully')
                return render_template('stat.html', channel_stats=channel_stats)
            else:
                logging.error('Channel not found or an error occurred during retrieval.')
                error_message = "Channel not found or an error occurred during retrieval."
                return render_template('stat.html', error_message=error_message)
            
    logging.info('No channel name provided for /stat')
    return render_template('stat.html', channel_stats=None)

def get_channel_stats(channel_name):
    youtube = build("youtube", "v3", developerKey=api_key)

    try:
        request = youtube.search().list(
            q=channel_name,
            type='channel',
            part='id',
            maxResults=1
        )
        response1 = request.execute()
        channel_id = response1['items'][0]['id']['channelId']
        
        request = youtube.channels().list(
            part='snippet,contentDetails,statistics',
            id=channel_id
        )
        response = request.execute()
        
        if 'items' in response:
            channel_info = response['items'][0]
            data = {
                'Channel_Name': channel_info['snippet']['title'],
                'Channel_ID': channel_info['id'],  # Ensure that 'Channel_ID' key is used
                'Subscribers': channel_info['statistics']['subscriberCount'],
                'Views': channel_info['statistics']['viewCount'],
                'Total_Videos': channel_info['statistics']['videoCount'],
                'playlist_id': channel_info['contentDetails']['relatedPlaylists']['uploads']
            }
            logging.info('Channel statistics retrieved successfully.')
            return data
        else:
            logging.error('Channel not found or an error occurred during retrieval.')
            return None
        
    except Exception as e:
        logging.error(f'An error occurred in get_channel_stats: {str(e)}')
        return None

@app.route('/likes', methods=['GET', 'POST'])
def get_video_details():
    download_link =''
    if request.method == 'POST':
        channel_name = request.form['channel_name']
        logging.info(f'Starting video details retrieval for channel: {channel_name}')
        # Call your function to get playlist ID
        channel_statistics = get_channel_stats(channel_name)
        if channel_statistics is not None:
            playlist_id = channel_statistics['playlist_id']

            if playlist_id:
                # Call your function to get video IDs
                logging.info('Getting video details for the playlist.')
                video_ids = get_videos_ids(playlist_id)  
                
                if video_ids:
                    logging.info('Video details retrieved successfully.')
                    video_details_df = get_video_details(video_ids)

                    if not video_details_df.empty:
                        # Save the video details to a CSV file
                        logging.info('Video details saved to CSV.')
                        filename = f"video_details.csv"
                        video_details_df.to_csv(filename, index=False, encoding='utf-8')

                        # Provide a link to download the file
                        download_link = f'File save as {filename}'
                        # return f"File generated successfully. {download_link}"

        # return "Failed to retrieve video details."
        logging.error('Failed to retrieve video details.')

    logging.info('No channel name provided for /likes')
    return render_template('likes.html', download_link=download_link)


# Function to get video IDs from a playlist
def get_videos_ids(playlist_id):
    logging.info('Getting video IDs...')
    youtube = build("youtube", "v3", developerKey=api_key)
    logging.info('Getting video IDs...')
    
    video_ids = []
    try:
        request = youtube.playlistItems().list(
            part='contentDetails',
            playlistId=playlist_id,
            maxResults=50)
        response = request.execute()
        for i in range(len(response['items'])):
            video_ids.append(response['items'][i]['contentDetails']['videoId'])
        next_page_token = response.get('nextPageToken')
        more_pages = True
        while more_pages:
            if next_page_token is None:
                more_pages = False
            else:
                request = youtube.playlistItems().list(
                    part='contentDetails',
                    playlistId=playlist_id,
                    maxResults=50,
                    pageToken=next_page_token)
                response = request.execute()
                for i in range(len(response['items'])):
                    video_ids.append(response['items'][i]['contentDetails']['videoId'])
                next_page_token = response.get('nextPageToken')
                
    except Exception as e:
        logging.error(f'An error occurred in get_videos_ids: {str(e)}')
    else:
        logging.info('Video IDs retrieval completed successfully.')
    
    return video_ids

# Function to get video details
def get_video_details(video_ids):
    youtube = build("youtube", "v3", developerKey=api_key)
    logging.info('Getting video details...')
    
    all_info = []
    video_details_df = pd.DataFrame()
    try:
        for i in range(0, len(video_ids), 50):
            request = youtube.videos().list(
                part="snippet,contentDetails,statistics",
                id=','.join(video_ids[i:i+50])
            )
            response = request.execute()
            for video in response['items']:
                keepstats = dict(Title=video['snippet']['title'],
                                 Published_date=video['snippet']['publishedAt'],
                                 Views=video['statistics'].get('viewCount', 0),
                                 Likes=video['statistics'].get('likeCount', 0),
                                 Comments=video['statistics'].get('commentCount', 0))
                all_info.append(keepstats)
    except Exception as e:
        logging.error(f'An error occurred in get_video_details: {str(e)}')
    else:
        logging.info('Video details retrieval completed successfully.')

    # Convert the list of dictionaries to a Pandas DataFrame
        video_details_df = pd.DataFrame(all_info)

    return video_details_df

@app.route('/predict', methods=['GET'])
def predict_form():
    logging.info('Accessed the /predict route')
    return render_template('predict.html')

filename = "<PATH>\video_details.csv"
df = pd.read_csv(filename)

logging.info('Loaded data from file: %s', filename)
# Split the data into training and testing sets
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Define the features and target variables
features = ['Views', 'Comments']
target = 'Likes'

logging.info('Features: %s', features)
logging.info('Target: %s', target)


# Train your models (XGBoost, KNN, and Random Forest)
xgb_model = xgb.XGBRegressor(objective='reg:squarederror',eval_metric='rmse', n_estimators=100, max_depth=3, learning_rate=0.1)
xgb_model.fit(train[features], train[target])
logging.info('XGBoost model trained')

knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(train[features], train[target])
logging.info('K-Nearest Neighbors model trained')


rf_model = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
rf_model.fit(train[features], train[target])
logging.info('Random Forest model trained')

# Create the voting regressor
voting_model = VotingRegressor([('xgb', xgb_model), ('rf', rf_model), ('knn', knn_model)])
voting_model.fit(train[features], train[target])
logging.info('Voting Regressor model trained')

@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        data = request.get_json()
        if data:
            views = float(data.get('Views', 0))
            comments = float(data.get('Comments', 0))
            test_predictions = voting_model.predict([[views, comments]])
            return jsonify({'prediction': float(test_predictions[0])})
        else:
            return jsonify({'error': 'JSON data is missing or invalid'}), 400
    else:
        return jsonify({'error': 'Unsupported Media Type: Content-Type should be application/json'}), 415

@app.route('/about')
def about():
    logging.info('Accessed the /about route')
    return render_template('about.html')

@app.route('/feedback')
def feedback():
    logging.info('Accessed the /feedback route')
    return render_template('feedback.html')

@app.route('/help')
def help():
    logging.info('Accessed the /help route')
    return render_template('help.html')

if __name__ == '__main__':
    logging.info('Starting the application.')
    app.run(debug=True)
