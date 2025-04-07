
# Prepare the data
lng = pd.read_csv('../Data/RealTime/iii. LNG.csv')
lng = rename_cols(lng)
lng = lng.rename(columns={'Close':'LNG Price','High':'High_LNG','Low':'Low_LNG','Open':'Open_LNG','Volume':'Volume_LNG'})
lng = lng.ffill()
data = lng['LNG Price'].values
n_steps = 7
train_data = data[:-n_steps]
test_data = data[-n_steps:]
train_X, train_y = [], []
for i in range(n_steps, len(train_data)):
    train_X.append(train_data[i - n_steps: i])
    train_y.append(train_data[i])
train_X = np.array(train_X).reshape(-1, n_steps, 1)
train_y = np.array(train_y)
test_X = np.array([test_data]).reshape(-1, n_steps, 1)
test_y = test_X
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(n_steps, 1), kernel_regularizer=regularizers.l2(0.02)))
model.add(Dropout(0.6))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit(train_X, train_y, epochs=100, batch_size=32, verbose=1, validation_data=(test_X, test_y), callbacks=[early_stopping])
predicted_y = model.predict(test_X).flatten()
mse = np.mean((test_data - predicted_y) ** 2)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

forecast_period = 7
forecast_X = np.array([test_data]).reshape(-1, n_steps, 1)
forecasted_values = []
for _ in range(forecast_period):
    forecast = model.predict(forecast_X).flatten().round(2)
    forecasted_values.append(forecast[0])
    forecast_X = np.roll(forecast_X, -1)
    forecast_X[-1, -n_steps:] = forecast
print("Forecasted Values:")
last_date = pd.to_datetime(lng['Date'].iloc[-1])
for i, forecast in enumerate(forecasted_values, 1):
    forecast_date = last_date + pd.DateOffset(days=i*1)
    forecast_date_str = forecast_date.strftime('%Y-%m-%d')
    print(f"{forecast_date_str}: {forecast}")
forecast_dates = [last_date + timedelta(days=i*1) for i in range(1, forecast_period + 1)]
forecast_df = pd.DataFrame({'Date': forecast_dates, 'LNG Price': forecasted_values})
forecast_df['Date'] = forecast_df['Date'].dt.strftime('%Y-%m-%d')
merged_df = pd.concat([lng, forecast_df], ignore_index=True)
merged_df['Date'] = pd.to_datetime(merged_df['Date'], dayfirst=False)
merged_df.set_index('Date', inplace=True)
merged_df['Difference'] = merged_df['LNG Price'].diff()
daily_index = pd.date_range(start=merged_df.index.min(), end=merged_df.index.max(), freq='D')
df_daily = pd.DataFrame(index=daily_index)
df_daily = df_daily.merge(merged_df, how='left', left_index=True, right_index=True)
df_daily=df_daily.dropna(subset=['LNG Price'])
df_daily.drop('Difference', axis=1, inplace=True)

df_daily = df_daily.round(2)
df_daily = df_daily.reset_index()
df_daily = df_daily.rename(columns={'index':'Date'})
df_daily['Date'] = pd.to_datetime(df_daily['Date']).dt.strftime('%Y-%m-%d')
df_daily.to_csv('../Data/2. LNG_Forecasted_Daily_LSTM.csv',index=False)
df_daily.to_csv('../Data/RealTime/iii. LNG_Forecasted_Daily_LSTM.csv',index=False)
print('-----------------------FORECAST COMPLETE-----------------------')
############################################# END OF FORECAST ############################################################

##################################################### FUNCTIONS FOR SS ######################################################################
def text_preparation(text):
    text = str(text)
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and digits using regular expression
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
    stop_words.update(['Ã','¢','â','¢'])
    tokens = [token for token in tokens if token not in stop_words]

    # Perform lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join tokens back to a preprocessed string
    preprocessed_text = " ".join(lemmatized_tokens)

    return preprocessed_text

##################################################### PRETRAIN MODELS #####################################################
# import torch
# from transformers import BertTokenizer, BertForSequenceClassification
# from nltk.sentiment import SentimentIntensityAnalyzer

# # Load pre-trained BERT model and tokenizer
# model_name = "bert-base-uncased"
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertForSequenceClassification.from_pretrained(model_name)

import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

##################################################### PRETRAIN MODELS #####################################################

# Function to read words from a text file and create a list
def read_words_from_file(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as file:
        return [word.strip() for word in file]

# Read positive and negative words from the text files with 'latin-1' encoding
positive_words = read_words_from_file('../Input Data/positive-words.txt', encoding='latin-1')
negative_words = read_words_from_file('../Input Data/negative-words.txt', encoding='latin-1')

# Function to calculate sentiment score with BERT
# def calculate_sentiment_score_with_bert(text):
#     # Tokenize the text and convert to tensors
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
#     # Forward pass through the model
#     outputs = model(**inputs)
#     logits = outputs.logits
#     probabilities = torch.sigmoid(logits)  # Apply sigmoid activation to get probabilities
#     sentiment_score = probabilities[:, 1].item()  # Get the probability of the positive class (index 1)

#     return sentiment_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Set model to evaluation mode

# Function to calculate sentiment scores in batches
def calculate_sentiment_score_with_bert(texts, batch_size=32):
    all_scores = []
    with torch.no_grad():  # Disable gradient calculation for faster inference
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            # Tokenize and move inputs to device
            inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            # Forward pass through the model
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.sigmoid(logits)  # Apply sigmoid activation
            # Collect probabilities for the positive class
            scores = probabilities[:, 1].tolist()
            all_scores.extend(scores)
    return all_scores

def get_sentiment_score(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)
    return sentiment_score['compound']

##################################################### END OF FUNCTIONS FOR SS ######################################################################

############################################# START OF EA ############################################################
# Get the current working directory
parent_directory = os.path.dirname(os.getcwd())
pdf_directory = os.path.join(parent_directory, 'Energy Aspects', 'PDFs')
print(pdf_directory)
output_csv_file = '../Data/eaoutput_new.csv'

def extract_all_content_from_pdf(pdf_file_path):
    with pdfplumber.open(pdf_file_path) as pdf:
        all_content = []
        for page_num in range(len(pdf.pages)):
            page = pdf.pages[page_num]
            page_content = {}

            # Extract text from the page
            page_text = page.extract_text()
            page_content['text'] = page_text

            # Extract tables from the page
            extracted_tables = page.extract_tables()
            if extracted_tables:
                page_content['tables'] = extracted_tables

            # Extract images from the page (if needed)
            # page_images = page.images
            # if page_images:
            #     page_content['images'] = page_images

            all_content.append(page_content)
        return all_content
    
 
# Function to get the creation date from the PDF metadata
def get_pdf_creation_date(pdf_file_path):
    with open(pdf_file_path, 'rb') as file:
        pdf = PyPDF2.PdfReader(file)
        info = pdf.metadata

        creation_date = info['/CreationDate']

        if creation_date:
            # Convert the creation date to datetime and return it
            return datetime.strptime(creation_date[2:16], "%Y%m%d%H%M%S")
        else:
            return None
        
def get_pdf_mod_date(pdf_file_path):
    with open(pdf_file_path, 'rb') as file:
        pdf = PyPDF2.PdfReader(file)
        info = pdf.metadata

        mod_date = info['/CreationDate']

        if mod_date:
            # Convert the creation date to datetime and return it
            return datetime.strptime(mod_date[2:16], "%Y%m%d%H%M%S")
        else:
            return None

def get_os_modified_date(file_path):
    modified_timestamp = os.path.getmtime(file_path)
    dt = datetime.fromtimestamp(modified_timestamp)
    return dt.strftime('%d/%m/%Y %H:%M')

# Subtract 14 days from the current time to get the threshold time
threshold_time = datetime.now() - timedelta(days=14)

# List all PDF files in the specified directory that are modified within the last 14 days
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf') and datetime.fromtimestamp(os.path.getmtime(os.path.join(pdf_directory, f))) > threshold_time]

# For all PDF files
# pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

# Prepare the header for the CSV file
header = ['PDF', 'Page', 'Text', 'Table', 'CreationDateTime','ModDateTime','OSModifiedDateTime']

# Write the extracted content to the CSV file
#30+s
with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile, escapechar='\\', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(header)
    # Loop through each PDF file and extract all content
    for pdf_file in pdf_files:
        pdf_file_path = os.path.join(pdf_directory, pdf_file)
        all_content = extract_all_content_from_pdf(pdf_file_path)
        creation_date = get_pdf_creation_date(pdf_file_path)
        mod_date = get_pdf_mod_date(pdf_file_path)
        os_modified_date = get_os_modified_date(pdf_file_path)


        # Write the extracted content for each PDF to the CSV file
        for page_num, page_content in enumerate(all_content, start=1):
            pdf_name = os.path.basename(pdf_file)
            page_text = page_content.get('text', '')
            tables = page_content.get('tables', [])
            for table_num, table_data in enumerate(tables, start=1):
                # Flatten the nested table_data into a comma-separated string
                flattened_table_data = [",".join(map(str, row)) for row in table_data]
                csv_writer.writerow([pdf_name, page_num, page_text, "|".join(flattened_table_data), creation_date,mod_date,os_modified_date])

ea = pd.read_csv('../Data/eaoutput_new.csv')
merged_data = ea.copy()
# Join 'Text' and 'Table' columns into a single 'Content' column
merged_data ['Content'] = merged_data['Text'].str.cat(merged_data['Table'], sep='\n')

grouped = merged_data.groupby('PDF')['Content'].apply('\n'.join).reset_index()
grouped.columns = ['PDF', 'PDFFullContent']

merged_data = merged_data.drop_duplicates(subset=['PDF','CreationDateTime'], keep='first').reset_index(drop=True)
merged_data = pd.merge(merged_data, grouped, on='PDF')
merged_data = merged_data.drop(columns=['Text','Table','Content','Page'])

merged_data['PDFFullContent'] = merged_data['PDFFullContent'].str.replace('\n', ' ')

merged_data['CreationDateTime'] = pd.to_datetime(merged_data['CreationDateTime'])
merged_data['ModDateTime'] = pd.to_datetime(merged_data['ModDateTime'])
merged_data['OSModifiedDateTime'] = pd.to_datetime(merged_data['OSModifiedDateTime'],dayfirst=True)

# Get PDFContentDate from top left corner of PDF
merged_data['PDFContentDate'] = merged_data['PDFFullContent'].str[:10] # Format: %d/%m/%Y
merged_data['PDFContentDate'] = pd.to_datetime(merged_data['PDFContentDate'], format='%d/%m/%Y', errors='coerce')

# Get PDFContentTime
merged_data['PDFContentDateTime'] = merged_data['PDFFullContent'].str[:16] # Format: %H:%M
merged_data['PDFContentDateTime'] = pd.to_datetime(merged_data['PDFContentDateTime'],errors='coerce')

# Get PDF Date from FileName
merged_data['PDFNameDate'] = merged_data['PDF'].str[:10] # Format: %Y-%m-%d
merged_data['PDFNameDate'] = pd.to_datetime(merged_data['PDFNameDate'], format='%Y-%m-%d', errors='coerce')

merged_data['CreationDate'] = merged_data['CreationDateTime'].dt.date
merged_data['OSModifiedDate'] = merged_data['OSModifiedDateTime'].dt.date

# Fill missing PDFContentDate with CreationDate
merged_data = merged_data[['PDF', 'PDFContentDateTime','PDFContentDate','PDFFullContent', 'PDFNameDate','OSModifiedDate','CreationDate', 'CreationDateTime','ModDateTime','OSModifiedDateTime']]

def replace_dates(row):
    # if PDFContentDate is not null, return PDFContentDate
    if pd.notnull(row['PDFContentDate']):
        return row['PDFContentDate']
    # if PDFContentDate is null, check if PDFNameDate is not null, return PDFNameDate
    elif (pd.isnull(row['PDFContentDate'])) & (pd.notnull(row['PDFNameDate'])):
        return row['PDFNameDate']
    # if PDFContentDate and PDFNameDate are null, check if OSModifiedDate is not null
    else:
        # if OSModifiedDate is null, return CreationDate, else check if OSModifiedDate and CreationDate are within 14 days of each other
        if (row['OSModifiedDate'] - row['CreationDate']) <= timedelta(days=14):
            return row['OSModifiedDate']
        else:
            return row['PDFContentDate']

merged_data['PDFContentDate'] = merged_data.apply(replace_dates, axis=1)
merged_data['PDFContentDate'] = pd.to_datetime(merged_data['PDFContentDate'], format='%Y-%m-%d', errors='coerce')

# Fill missing PDFContentDateTime with CreationDateTime
merged_data['PDFContentDateTime'] = merged_data['PDFContentDateTime'].fillna(merged_data['CreationDateTime'])

# Convert OSModifiedDate, CreationDate to datetime
merged_data['OSModifiedDate'] = pd.to_datetime(merged_data['OSModifiedDate'], format='%Y-%m-%d', errors='coerce')
merged_data['CreationDate'] = pd.to_datetime(merged_data['CreationDate'], format='%Y-%m-%d', errors='coerce')

merged_data.drop(columns=['PDFNameDate'],inplace=True) # Drop unnecessary columns
merged_data.rename(columns={'PDFFullContent':'Content'},inplace=True) # Rename column

merged_data.sort_values(by=['PDFContentDateTime'],ascending=True,inplace=True)
merged_data = merged_data[['PDFContentDateTime','PDFContentDate','PDF','Content', 'OSModifiedDate','CreationDate', 'CreationDateTime','ModDateTime','OSModifiedDateTime']].reset_index(drop=True)

merged_data.to_excel('../Data/Check/eaoutput_cleaned.xlsx', index=False) # For Checking Purposes

##################################################### GENERATE SENTIMENT SCORE FOR EA ######################################################################
print("--------------------------------- Cleaning Inputs for EA ---------------------------------")

# Assuming you have a DataFrame named 'df' with a column 'News' containing text data
merged_data['Content'] = merged_data['Content'].apply(lambda x: text_preparation(x))

print("--------------------------------- Calculating EA Sentiment Scores ---------------------------------")
df2 = merged_data.copy()
df2['EA_SS_1'] = df2['Content'].apply(lambda x: get_sentiment_score(x))
# Get sentiment scores for each text
# df2['EA_SS_3'] = df2['Content'].apply(lambda x: calculate_sentiment_score_with_bert(x))
df2['EA_SS_3'] = calculate_sentiment_score_with_bert(df2['Content'].tolist())

df2.to_excel('../Data/Roberta_EAwSS_recent.xlsx',index=False)

existing_eapdfs = pd.read_excel('../Data/Roberta_EAwSS_new.xlsx')
df3 = pd.concat([existing_eapdfs,df2],axis=0).copy()
df3.sort_values(by=['PDFContentDateTime'],ascending=True,inplace=True)
df3 = df3.drop_duplicates(subset=['PDF','Content'],keep='first').reset_index(drop=True)
df3.to_excel('../Data/Roberta_EAwSS_new.xlsx',index=False)

# Subtract 16 hours from 'PDFContentDateTime' to make days start at 4pm
########################## Explanation: The subtraction of 16 hours is done to shift the start of the day to 4 PM of the previous day. 
# In Python's pandas library, when you resample a time-series data to a daily frequency using 'D', the day is considered to start at midnight (00:00) by default. 
# However, in your case, you want the day to start at 4 PM. To achieve this, 16 hours are subtracted from each timestamp. This effectively shifts the start of the day to 4 PM of the previous day. 
# So, when you resample the data to a daily frequency, the data from 4 PM of one day to 3:59 PM of the next day is considered as one day. This is why 16 hours are subtracted.
df3['PDFContentDateTime'] = df3['PDFContentDateTime'] - pd.Timedelta(hours=16)
df3.set_index('PDFContentDateTime', inplace=True)

# Resample and calculate the mean
df3_resampled = df3[['EA_SS_1', 'EA_SS_3']].resample('D').mean()

# Convert the index back to the original datetime
df3_resampled.index = df3_resampled.index + pd.Timedelta(hours=16)

# Add dates til today
today = datetime.today().date()
idx = pd.date_range(df3_resampled.index.min(), today)
df3_resampled = df3_resampled.reindex(idx, fill_value=np.nan)

# Replace Date with actual date of prediction
################################################### Note: 25/1/2024 16:00 = 26/1/2024's data
df3_resampled['Date'] = (df3_resampled.index + pd.Timedelta(hours=16)).date
df3_resampled['StartDateTime'] = df3_resampled.index
df3_resampled.reset_index(drop=True, inplace=True)
df3_resampled['Date'] = pd.to_datetime(df3_resampled['Date'])
df3_resampled.set_index('Date', inplace=True)
df3_weekdays = df3_resampled[df3_resampled.index.dayofweek < 5] # Drop weekends based on actual date of prediction

####################################### DAILY #######################################
# Calculate the exponential moving average of past 5 days, excluding StartDateTime
df3 = df3_weekdays.drop(columns=['StartDateTime']).ewm(span=5, adjust=False).mean()
df3['StartDateTime'] = df3_weekdays['StartDateTime']
df3.to_csv('../Data/Check/Roberta_EAwSS_new_daily.csv',index=True)

####################################### WEEKLY #######################################
# Resample Weekly Average and calculate the mean stored in new columns EA_WA_1 and EA_WA_3 
df3_weekly = df3[['EA_SS_1', 'EA_SS_3']].resample('W').mean()
df3_weekly.rename(columns={'EA_SS_1':'EA_WA_1','EA_SS_3':'EA_WA_3'},inplace=True)
df3_weekly.to_csv('../Data/Check/Roberta_EAwSS_new_weekly.csv',index=True)

####################################### COMBINE DAILY+WEEKLY #######################################
df3_combined = pd.concat([df3, df3_weekly], axis=1).bfill()

# Drop weekends - 5: Saturday, 6: Sunday
df3_combined = df3_combined[df3_combined.index.dayofweek < 5]
df3_combined.to_csv('../Data/Check/Roberta_EAwSS_new_combined.csv',index=True)

ea_daily = df3_combined[['EA_SS_1','EA_SS_3','EA_WA_1','EA_WA_3']].reset_index(drop=False).copy()
ea_daily.to_csv('../Data/9b. Roberta_EAPDF_SS_Daily.csv',index=False)
print('EA Sentiment Scores Calculated')

############################################# END OF EA ############################################################

##################################################### START OF FGE ######################################################################
def process_fgepdf():
    fgepdf_directory = os.path.join(parent_directory, 'FGE')
        
    # Subtract 14 days from the current time to get the threshold time
    threshold_time = datetime.now() - timedelta(days=14)

    # List all PDF files in the specified directory that are modified within the last 14 days
    pdf_files = [f for f in os.listdir(fgepdf_directory) if f.endswith('.pdf') and datetime.fromtimestamp(os.path.getmtime(os.path.join(fgepdf_directory, f))) > threshold_time]
    # pdf_files = [f for f in os.listdir(fgepdf_directory) if f.endswith('.pdf')]

    def extract_date_from_filename(file_name):
        # Define patterns to match different date formats
        date_patterns = [
            r'\d{1,2} \w+ \d{4}',  # Matches "20 March 2023"
            r'\d{4}-\d{2}-\d{2}',  # Matches "2023-03-20"
            r'\d{2}-\d{2}-\d{4}',  # Matches "20-03-2023"
        ]

        for pattern in date_patterns:
            date_match = re.search(pattern, file_name)
            if date_match:
                try:
                    # Try to parse the extracted date string
                    date_str = date_match.group(0)
                    for fmt in ('%d %B %Y', '%Y-%m-%d', '%d-%m-%Y'):
                        try:
                            date_obj = datetime.strptime(date_str, fmt)
                            return date_obj.strftime('%Y-%m-%d')
                        except ValueError:
                            pass
                except ValueError:
                    pass

        return None

    # Prepare the header for the CSV file
    def extract_date_from_text(text):
        # Define patterns to match different date formats within the text
        date_patterns = [
            r'\d{1,2} \w+ \d{4}',  # Matches "20 March 2023"
            r'\d{4}-\d{2}-\d{2}',  # Matches "2023-03-20"
            r'\d{2}-\d{2}-\d{4}',  # Matches "20-03-2023"
        ]

        for pattern in date_patterns:
            date_match = re.search(pattern, text)
            if date_match:
                try:
                    # Try to parse the extracted date string
                    date_str = date_match.group(0)
                    for fmt in ('%d %B %Y', '%Y-%m-%d', '%d-%m-%Y'):
                        try:
                            date_obj = datetime.strptime(date_str, fmt)
                            return date_obj.strftime('%Y-%m-%d')
                        except ValueError:
                            pass
                except ValueError:
                    pass

        return None

    def parse_pdf_text(text, file_name, file_path):
        date_str = extract_date_from_filename(file_name)
        if not date_str:
            date_str = extract_date_from_text(text)
        if not date_str:
            date_str = datetime.now().strftime('%Y-%m-%d')

        time_str = extract_time_from_file_metadata(file_path)
        date_time_str = f"{date_str} {time_str}"

        lines = text.split('\n')
        content = "\n".join(lines)  # Preserve line breaks for better readability
        tasks = {f"Task{i+1}": line for i, line in enumerate(lines) if "Task" in line}  # Example task extraction logic

        return {
            "FGEDateTime": date_time_str,
            "Date": date_str,
            "Time": time_str,
            "Content": content,
            **tasks
        }

    def extract_time_from_file_metadata(file_path):
        # Get the last modification time from the file metadata
        timestamp = os.path.getmtime(file_path)
        time_str = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
        return time_str

    header = ['PDF', 'Page', 'Text', 'Table', 'FileDate','CreationDate','CreationDateTime','ModDateTime','OSModifiedDateTime']
    with open('../Data/fgeoutput_newtest.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile, escapechar='\\', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(header)
        # Loop through each PDF file and extract all content
        for pdf_file in pdf_files:
            pdf_file_path = os.path.join(fgepdf_directory, pdf_file)
            all_content = extract_all_content_from_pdf(pdf_file_path)
            
            date_str = extract_date_from_filename(pdf_file)
            if not date_str:
                date_str = extract_date_from_text(all_content)
            if not date_str:
                date_str = datetime.now().strftime('%Y-%m-%d')

            time_str = extract_time_from_file_metadata(pdf_file_path)
            datefromfilename = f"{date_str} {time_str}"
            
            creation_date = get_pdf_creation_date(pdf_file_path)
            mod_date = get_pdf_mod_date(pdf_file_path)
            os_modified_date = get_os_modified_date(pdf_file_path)


            # Write the extracted content for each PDF to the CSV file
            for page_num, page_content in enumerate(all_content, start=1):
                pdf_name = os.path.basename(pdf_file)
                page_text = page_content.get('text', '')
                tables = page_content.get('tables', [])
                for table_num, table_data in enumerate(tables, start=1):
                    # Flatten the nested table_data into a comma-separated string
                    flattened_table_data = [",".join(map(str, row)) for row in table_data]
                    csv_writer.writerow([pdf_name, page_num, page_text, "|".join(flattened_table_data), datefromfilename, creation_date,mod_date,os_modified_date])
                    
                    
    fge = pd.read_csv('../Data/fgeoutput_newtest.csv')
    fge['CreationDate'] = pd.to_datetime(fge['CreationDateTime']).dt.date
    fge = fge.drop(columns=['CreationDateTime'])
    fge = fge.rename(columns={'FileDate':'CreationDateTime'})

    merged_data = fge.copy()
    # Join 'Text' and 'Table' columns into a single 'Content' column
    merged_data ['Content'] = merged_data['Text'].str.cat(merged_data['Table'], sep='\n')

    grouped = merged_data.groupby('PDF')['Content'].apply('\n'.join).reset_index()
    grouped.columns = ['PDF', 'PDFFullContent']

    merged_data = merged_data.drop_duplicates(subset=['PDF','CreationDateTime'], keep='first').reset_index(drop=True)
    merged_data = pd.merge(merged_data, grouped, on='PDF')
    merged_data = merged_data.drop(columns=['Text','Table','Content','Page'])

    merged_data['PDFFullContent'] = merged_data['PDFFullContent'].str.replace('\n', ' ')

    merged_data['CreationDateTime'] = pd.to_datetime(merged_data['CreationDateTime'], format='mixed')
    merged_data['ModDateTime'] = pd.to_datetime(merged_data['ModDateTime'], format='mixed')
    merged_data['OSModifiedDateTime'] = pd.to_datetime(merged_data['OSModifiedDateTime'], format='mixed')

    # Get PDFContentTime
    merged_data['PDFContentDateTime'] = merged_data['PDFFullContent'].str[:16] # Format: %H:%M
    merged_data['PDFContentDateTime'] = pd.to_datetime(merged_data['PDFContentDateTime'],errors='coerce')

    # Get PDF Date from FileName
    merged_data['PDFNameDate'] = merged_data['PDF'].str[:10] # Format: %Y-%m-%d
    merged_data['PDFNameDate'] = pd.to_datetime(merged_data['PDFNameDate'], format='%Y-%m-%d', errors='coerce')

    merged_data['CreationDate'] = merged_data['CreationDateTime'].dt.date
    merged_data['OSModifiedDate'] = merged_data['OSModifiedDateTime'].dt.date
    merged_data['PDFContentDate'] = merged_data['CreationDate']
    merged_data['PDFContentDate'] = pd.to_datetime(merged_data['PDFContentDate'], format='%Y-%m-%d', errors='coerce')

    # Fill missing PDFContentDate with CreationDate
    merged_data = merged_data[['PDF', 'PDFContentDateTime','PDFContentDate','PDFFullContent', 'PDFNameDate','OSModifiedDate','CreationDate', 'CreationDateTime','ModDateTime','OSModifiedDateTime']]


    # # Fill missing PDFContentDateTime with CreationDateTime
    merged_data['PDFContentDateTime'] = merged_data['PDFContentDateTime'].fillna(merged_data['CreationDateTime'])

    # Convert OSModifiedDate, CreationDate to datetime
    merged_data['OSModifiedDate'] = pd.to_datetime(merged_data['OSModifiedDate'], format='%Y-%m-%d', errors='coerce')
    merged_data['CreationDate'] = pd.to_datetime(merged_data['CreationDate'], format='%Y-%m-%d', errors='coerce')

    merged_data.drop(columns=['PDFNameDate'],inplace=True) # Drop unnecessary columns
    merged_data.rename(columns={'PDFFullContent':'Content'},inplace=True) # Rename column

    merged_data.sort_values(by=['PDFContentDateTime'],ascending=True,inplace=True)
    merged_data = merged_data[['PDFContentDateTime','PDFContentDate','PDF','Content', 'OSModifiedDate','CreationDate', 'CreationDateTime','ModDateTime','OSModifiedDateTime']].reset_index(drop=True)

    merged_data.to_excel('../Data/Check/fgeoutput_cleaned.xlsx', index=False) # For Checking Purposes

    ##################################################### GENERATE SENTIMENT SCORE FOR FGE ######################################################################
    print("--------------------------------- Cleaning Inputs for FGE ---------------------------------")

    # Assuming you have a DataFrame named 'df' with a column 'News' containing text data
    merged_data['Content'] = merged_data['Content'].apply(lambda x: text_preparation(x))

    def clean_content(content):
        # Remove repetitive or unwanted text
        unwanted_phrases = [
            "www.fgenergy.com",
            r'Page \d+ of \d+',
            "Chairman’s Corner",
            r"© \d{4} FGE.*",  # This will match "© 2024 FGE", "© 2023 FGE", etc., and anything that follows
            "For queries, please email FGE@fgenergy.com",
            "The dissemination, distribution, or copying by any means whatsoever without FGE’s prior written consent is strictly prohibited."
            r"If you have any questions regarding this presentation, please contact us at.*"  # This will match the specified phrase and anything that follows
        ]
        for phrase in unwanted_phrases:
            content = re.sub(phrase, '', content, flags=re.IGNORECASE)
        # Remove excessive newlines and whitespace
        content = re.sub(r'\n\s*\n', '\n', content)
        # Remove non-ASCII characters, punctuations
        content = re.sub(r'[^\x00-\x7F]+', '', content)
        content = re.sub(r'[^\w\s]', '', content)
        # Remove Fereidun Fesharaki  
        content = re.sub(r'Fereidun Fesharaki', '', content)
        # Strip leading/trailing whitespace
        content = content.strip()
        return content

    merged_data['Content'] = merged_data['Content'].apply(clean_content)

    print("--------------------------------- Calculating FGE Sentiment Scores ---------------------------------")
    df2 = merged_data.copy()
    df2['FGE_SS_1'] = df2['Content'].apply(lambda x: get_sentiment_score(x))
    # Get sentiment scores for each text
    # df2['FGE_SS_3'] = df2['Content'].apply(lambda x: calculate_sentiment_score_with_bert(x))
    df2['FGE_SS_3'] = calculate_sentiment_score_with_bert(df2['Content'].tolist())

    df2.to_excel('../Data/FGEwSS_recent.xlsx',index=False)

    existing_eapdfs = pd.read_excel('../Data/FGEwSS_new.xlsx')
    df3 = pd.concat([existing_eapdfs,df2],axis=0).copy()
    df3.sort_values(by=['PDFContentDateTime'],ascending=True,inplace=True)
    df3 = df3.drop_duplicates(subset=['PDF','Content'],keep='first').reset_index(drop=True)
    df3.to_excel('../Data/FGEwSS_new.xlsx',index=False)
    df3['PDFContentDateTime'] = pd.to_datetime(df3['PDFContentDateTime'])
    # Subtract 16 hours from 'PDFContentDateTime' to make days start at 4pm
    ########################## Explanation: The subtraction of 16 hours is done to shift the start of the day to 4 PM of the previous day. 
    # In Python's pandas library, when you resample a time-series data to a daily frequency using 'D', the day is considered to start at midnight (00:00) by default. 
    # However, in your case, you want the day to start at 4 PM. To achieve this, 16 hours are subtracted from each timestamp. This effectively shifts the start of the day to 4 PM of the previous day. 
    # So, when you resample the data to a daily frequency, the data from 4 PM of one day to 3:59 PM of the next day is considered as one day. This is why 16 hours are subtracted.
    df3['PDFContentDateTime'] = df3['PDFContentDateTime'] - pd.Timedelta(hours=16)
    df3.set_index('PDFContentDateTime', inplace=True)
    
    ea_daily = pd.read_csv('../Data/9. EAPDF_SS_Daily.csv').copy()

    ea_daily['Date'] = pd.to_datetime(ea_daily['Date'])
    ea_daily.set_index('Date', inplace=True)

    fgedf3_weekly = df3[['FGE_SS_1', 'FGE_SS_3']].resample('W').mean()
    fgedf3_weekly = fgedf3_weekly.ffill()
    fgedf3_weekly.rename(columns={'FGE_SS_1':'FGE_WA_1','FGE_SS_3':'FGE_WA_3'},inplace=True)
    fgedf3_daily = fgedf3_weekly.resample('D').mean().bfill()
    fgedf3_daily = fgedf3_daily.reindex(ea_daily.index)
    fgedf3_daily = fgedf3_daily.bfill().ffill()
    fgedf3_daily.reset_index(drop=False, inplace=True)
    fgedf3_daily.to_csv('../Data/10. FGEPDF_SS_Daily.csv',index=False)
    print('FGE Daily Sentiment Score Calculated')
process_fgepdf()
##################################################### END OF FGE ############################################################

##################################################### START OF DATA PREPARATION ############################################################
def merge_input(main, merge): ## Function to merge by Date
    if 'Date' in main.columns and main['Date'].dtype != 'datetime64[ns]':
        main['Date'] = pd.to_datetime(main['Date'], format='mixed')
    if 'Date' in merge.columns and merge['Date'].dtype != 'datetime64[ns]':
        merge['Date'] = pd.to_datetime(merge['Date'], format='mixed')

    merged_df = pd.merge(main, merge, on='Date', how='left')
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])  # Convert 'Date' column to datetime type
    merged_df = merged_df.sort_values(by='Date')
    merged_df = merged_df.ffill()  # Forward-fill missing values
    return merged_df

def feature(mergeddf,sd,ed): ## Function for feature extraction
    mergeddf = mergeddf.sort_values(by='Date')
    mergeddf['lag1'] = mergeddf['Close'].shift(1)
    mergeddf['lag2'] = mergeddf['Close'].shift(2)
    mergeddf['lag3'] = mergeddf['Close'].shift(3)
    mergeddf['lag4'] = mergeddf['Close'].shift(4)
    mergeddf['lag5'] = mergeddf['Close'].shift(5)
    mergeddf['lagged3'] = (mergeddf['lag1']+mergeddf['lag2']+mergeddf['lag3'])/3
    mergeddf['lagged5'] = (mergeddf['lag1']+mergeddf['lag2']+mergeddf['lag3']+mergeddf['lag4']+mergeddf['lag5'])/5

    mergeddf['MA_10'] = mergeddf['Close'].rolling(window=10).mean()
    mergeddf['MA_20'] = mergeddf['Close'].rolling(window=20).mean()
    mergeddf['MA_30'] = mergeddf['Close'].rolling(window=30).mean()
    mergeddf['MA_60'] = mergeddf['Close'].rolling(window=60).mean()
    mergeddf['MA_250'] = mergeddf['Close'].rolling(window=250).mean()
    mergeddf['long_shortma'] = mergeddf['MA_250'] - mergeddf['MA_60']

    # Create a date range for filtering of data
    start_date = pd.to_datetime(sd, format='%d-%m-%Y')
    end_date = pd.to_datetime(ed, format='%d-%m-%Y')

    mergeddf['Date'] = pd.to_datetime(mergeddf['Date'])  # Convert 'Date' column to datetime type
    mergeddf = mergeddf[(mergeddf['Date'] >= start_date) & (mergeddf['Date'] <= end_date)]

    mergeddf = mergeddf.set_index('Date')
    mergeddf['Close'] = mergeddf.pop('Close')
    mergeddf = mergeddf.sort_values(by='Date')
    mergeddf = mergeddf.ffill()

    return mergeddf.dropna()

start_date = datetime(2021, 1, 1)
start_date = start_date.strftime('%d-%m-%Y')



##################################################### START OF DATA PREPARATION ############################################################
def merge_input(main, merge): ## Function to merge by Date
    if 'Date' in main.columns and main['Date'].dtype != 'datetime64[ns]':
        main['Date'] = pd.to_datetime(main['Date'], format='mixed')
    if 'Date' in merge.columns and merge['Date'].dtype != 'datetime64[ns]':
        merge['Date'] = pd.to_datetime(merge['Date'], format='mixed')

    merged_df = pd.merge(main, merge, on='Date', how='left')
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])  # Convert 'Date' column to datetime type
    merged_df = merged_df.sort_values(by='Date')
    merged_df = merged_df.ffill()  # Forward-fill missing values
    return merged_df

def feature(mergeddf,sd,ed): ## Function for feature extraction
    
    #Brent Price
    mergeddf = mergeddf.sort_values(by='Date')
    mergeddf['lag1'] = mergeddf['Close'].shift(1)
    mergeddf['lag2'] = mergeddf['Close'].shift(2)
    mergeddf['lag3'] = mergeddf['Close'].shift(3)
    mergeddf['lag4'] = mergeddf['Close'].shift(4)
    mergeddf['lag5'] = mergeddf['Close'].shift(5)
    mergeddf['lagged3'] = (mergeddf['lag1']+mergeddf['lag2']+mergeddf['lag3'])/3
    mergeddf['lagged5'] = (mergeddf['lag1']+mergeddf['lag2']+mergeddf['lag3']+mergeddf['lag4']+mergeddf['lag5'])/5

    mergeddf['MA_10'] = mergeddf['Close'].rolling(window=10).mean()
    mergeddf['MA_20'] = mergeddf['Close'].rolling(window=20).mean()
    mergeddf['MA_30'] = mergeddf['Close'].rolling(window=30).mean()
    mergeddf['MA_60'] = mergeddf['Close'].rolling(window=60).mean()
    mergeddf['MA_250'] = mergeddf['Close'].rolling(window=250).mean()
    mergeddf['long_shortma'] = mergeddf['MA_250'] - mergeddf['MA_60']

    # LNG
    mergeddf['LNG_MA_20'] = mergeddf['LNG Price'].rolling(window=20).mean()
    mergeddf['LNG_MA_60'] = mergeddf['LNG Price'].rolling(window=60).mean()
    mergeddf['LNG_MA_250'] = mergeddf['LNG Price'].rolling(window=250).mean()
    mergeddf['LNG_long_shortma'] = mergeddf['LNG_MA_250'] - mergeddf['LNG_MA_60']
    
    # DXY
    mergeddf['DXY_MA_20'] = mergeddf['DXY'].rolling(window=20).mean()
    mergeddf['DXY_MA_60'] = mergeddf['DXY'].rolling(window=60).mean()
    mergeddf['DXY_MA_250'] = mergeddf['DXY'].rolling(window=250).mean()
    mergeddf['DXY_long_shortma'] = mergeddf['DXY_MA_250'] - mergeddf['DXY_MA_60']    

    start_date = pd.to_datetime(sd, format='%d-%m-%Y')
    end_date = pd.to_datetime(ed, format='%d-%m-%Y')

    mergeddf['Date'] = pd.to_datetime(mergeddf['Date'])  # Convert 'Date' column to datetime type
    mergeddf = mergeddf[(mergeddf['Date'] >= start_date) & (mergeddf['Date'] <= end_date)]

    mergeddf = mergeddf.set_index('Date')
    mergeddf['Close'] = mergeddf.pop('Close')
    mergeddf = mergeddf.sort_values(by='Date')
    mergeddf = mergeddf.ffill()

    return mergeddf


start_date = datetime(2021, 1, 1)
start_date = start_date.strftime('%d-%m-%Y')
bp = bp[['Date','Close','Open','High','Low','Volume','Contract']].copy()
if bp['Contract'].iloc[-1] == 'XILL001':
    pass
else:
    bp = nextmonth[['Date','Close','Open','High','Low','Volume']].copy()   
     
m1 = merge_input(bp,dxy)
m2 = merge_input(m1, ovx)
m3 = merge_input(m2, lng)
# # if hour is before 10am sgt,
# if datetime.now().hour < 10:
#     tradingday = bp['Date'].iloc[-1] + timedelta(days=2)
# else:
#     tradingday = bp['Date'].iloc[-1] + timedelta(days=1)

tradingday = tradingday.strftime('%d-%m-%Y')
print(tradingday,"=====================")
df2 = feature(m3,start_date,tradingday)

df2.reset_index(drop=False).to_csv('../Data/V7PredictiveData_woSS.csv',index=False)

print("-------------------- Read Datasets to Retrieve the Daily and Weekly SSs --------------------")
eapdfs = pd.read_csv('../Data/9. EAPDF_SS_Daily.csv').copy()
eapdfs['Date']=pd.to_datetime(eapdfs['Date'],format='mixed')

fgepdfs = pd.read_csv('../Data/10. FGEPDF_SS_Daily.csv').copy()
fgepdfs['Date']=pd.to_datetime(fgepdfs['Date'],format='mixed')

def prep_pdfs(df):
    # expand date range of df till today
    today = datetime.today().date()
    idx = pd.date_range(df['Date'].min(), today)
    # exclude weekends
    idx = idx[idx.dayofweek < 5]
    df = df.set_index('Date').reindex(idx, fill_value=np.nan).rename_axis('Date').reset_index()
    # Apply the ewm function only where the values are NaN
    df = df.apply(lambda x: x.ewm(span=5, adjust=False).mean() if np.isnan(x).any() else x)
    return df

# Now you can use this function to process both dataframes
eapdfs = prep_pdfs(eapdfs)
fgepdfs = prep_pdfs(fgepdfs)

print("-------------------- Merging Datasets --------------------")

dfwoss = pd.read_csv('../Data/V7PredictiveData_woSS.csv')
dfwoss['Date']=pd.to_datetime(dfwoss['Date'])

# volumeafter = dfwoss['Volume'].values[-1]
# volumechange = volumeafter - volumeprev

dfss = pd.merge(dfwoss, eapdfs, on='Date').copy()
dfss = pd.merge(dfss, fgepdfs, on='Date').copy()
dfss['Close'] = dfss.pop('Close')

dfss.to_csv('../Data/V7PredictiveData_wFGEEASS.csv',index=False)
print("-------------------- Predictive Data with EA and FGE SSs Saved --------------------")

############################################# END OF DATA PREP ############################################################


##################################################### START OF MODEL ############################################################
def printformat(title,next_day_prediction,predictionpricechange,predicteddirection):
    print(f"\033[1m\033[4mModel {title}\033[0m")
    print(f"Predicted Price: $ {next_day_prediction.values[0]}")
    print(f"Predicted Price Change: $ {predictionpricechange}")
    print(f"Predicted Direction:  {predicteddirection}\n")
    
    
def output(data): ## For Viz
    # Generate buy and sell signals
    data['buy_sell_signal'] = np.where(data['long_shortma'] > 0, 1, -1)                                                                  # Encode buy/sell signals
        
    data['Predicted Price'] = data['Predicted Close']
    data['Previous Close - Actual'] = data['Actual Close'].shift(1)
    price_change_prev = data['Actual Close'] - data['Previous Close - Actual']                                                          # Actual Price Change
    price_change_prev_model =  data['Predicted Price'] - data['Previous Close - Actual']                                                # Price Change based on Prediction
    predicteddirection = ['Increase' if change >= 0 else 'Decrease' for change in price_change_prev_model]                              # Predicted Direction    
    actualdirection = ['Increase' if achange >= 0 else 'Decrease' for achange in price_change_prev]                                     # Direction    
    alignment = ['Y' if change * prev_change > 0 else 'N' for change, prev_change in zip(price_change_prev_model, price_change_prev)]   # Alignment
    percent_error = ((data['Actual Close'] - data['Predicted Price']) / data['Predicted Price']) * 100                                  # % Error

    data['Price Change from Previous'] = price_change_prev.round(2)
    data['Price Change'] = price_change_prev_model.round(2)
    data['Predicted Direction'] = predicteddirection
    data['Actual Direction'] = actualdirection
    data['B/S @ Predicted'] = np.where(data['Actual Direction'] == 'Decrease', 'Sell', 'Buy')
    data['Alignment'] = alignment
    data['% Error'] = percent_error.round(2)
    
    ###################################################### PnL Calculation ######################################################
    pnl_values = []

    # Iterate through each row in data
    for index, row in data.iterrows():
        if row['Actual Direction'] == 'Increase':
            if row['Alignment'] == 'N':
                pnl = (row['Actual Close'] - row['Previous Close - Actual'])
            else:
                pnl = (row['Predicted Price'] - row['Previous Close - Actual'])
        elif row['Actual Direction'] == 'Decrease':
            if row['Alignment'] == 'N':
                pnl = (row['Previous Close - Actual'] - row['Actual Close'])
            else:
                pnl = row['Previous Close - Actual'] - row['Predicted Price']          

        pnl_values.append(pnl)

    # Assign the list of PnL values to the 'PnL' column in data
    data['PnL'] = pnl_values
    data['PnL'] = data['PnL'].round(4)

    ################################################## End of PnL Calculation ##################################################
    
    cols_order = ['Date', 'Actual Close', 'Predicted Price','B/S @ Predicted', 'PnL',
                  'Previous Close - Actual', 'Predicted Direction','Actual Direction', 'Alignment', 'Price Change from Previous',
                  'Price Change', '% Error','buy_sell_signal']

    data = data[cols_order]
    print((data['Alignment'].value_counts(normalize=True) * 100).round(2).astype(str) + '%')
    accuracy = (data['Alignment'].value_counts(normalize=True) * 100).round(2).astype(str) + '%'
    return data[1:],accuracy

def get_viz(df,plotname,modelname):

    data = df.copy()
    data['Date'] = pd.to_datetime(data['Date'],dayfirst=True)

    # Encode 'Y' and 'N' as 1 and 0
    data['Alignment'] = data['Alignment'].map({'Y': 1, 'N': 0})
    data['Predicted Direction'] = data['Predicted Direction'].map({'Increase': 1, 'Decrease': 0})
    data['Actual Direction'] = data['Actual Direction'].map({'Increase': 1, 'Decrease': 0})

    # Get the predicted and actual labels
    predicted_labels = data['Predicted Direction']
    actual_labels = data['Actual Direction']

    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(actual_labels, predicted_labels)

    # Calculate the AUC score
    auc_score = auc(fpr, tpr)
    # Calculate the KS score
    ks_score = max(tpr - fpr)

    # Print the results
    print('AUC score:', auc_score, ' KS score:', ks_score)

    # Calculate the confusion matrix and classification report
    cm = confusion_matrix(actual_labels, predicted_labels)
    report = classification_report(actual_labels, predicted_labels)
    
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create a heatmap of the confusion matrix
    im = ax.imshow(cm, cmap='Blues')
    # Add labels to the heatmap
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('Actual label')
    ax.set_title('Confusion matrix')

    # Add a colorbar to the heatmap
    fig.colorbar(im)

    # Add text annotations to the heatmap
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]:.1f}', ha='center', va='center', color='white')

    # Rotate the x labels to prevent overlapping
    plt.xticks(rotation=45)    

    # Calculate the classification report
    report = classification_report(actual_labels, predicted_labels)
    
    # Print the classification report
    print(report)
      

# Define a function to calculate feature importance by mean decrease accuracy
def calculate_feature_importance(model, X, y):
    baseline_score = model.evaluate(X, y)
    importance_scores = np.zeros(X.shape[1])

    for feature in range(X.shape[1]):
        X_shuffled = X.copy()
        np.random.shuffle(X_shuffled[:, feature])
        shuffled_score = model.evaluate(X_shuffled, y)
        importance_scores[feature] = baseline_score - shuffled_score

    return importance_scores

def predict(df2,splitind,eta, max_depth, min_child_weight,plotname,modelname):
    ################################ Data Split ################################
    # df2['Close'].iloc[-1] = df2['MA_20'].iloc[-1]
    # df2['Close'].iloc[-1] = 82.76
    # df2['LNG Price'].iloc[-1] = 2.17
    # df2 = df2.drop(columns=['MA_20'])
    # df2['Close'].iloc[-2] = 74.74 # temp fix for every 1st trading day of the month

    # Split the data into train and test sets
    split_index = int(splitind* len(df2))
    train = df2.iloc[:split_index]
    test = df2.iloc[split_index:]

    train_ma60 = train['MA_60']
    train_ma250 = train['MA_250']
    train = train.drop(columns=['MA_60','MA_250'])

    test_ma60 = test['MA_60']
    test_ma250 = test['MA_250']
    test = test.drop(columns=['MA_60','MA_250'])

    # Correlation Matrix
    df3 = df2.drop(columns=['MA_60','MA_250'])
    plt.figure(figsize=(8, 4))
    mask = np.triu(np.ones_like(df3.corr(), dtype=bool))
    sns.heatmap(df3.corr(),mask=mask, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    # plt.show() 

    lambda_list = []  # Initialize the lambda list

    scalers = {}
    train_scaled = np.zeros_like(train.values)
    test_scaled = np.zeros_like(test.values)
    features = [col for col in train.columns if col != 'Close']

    for i, feature in enumerate(features):
        print(feature)
        scaler = MinMaxScaler()
        train_scaled[:, i] = scaler.fit_transform(train.iloc[:, i].values.reshape(-1, 1)).flatten()
        test_scaled[:, i] = scaler.transform(test.iloc[:, i].values.reshape(-1, 1)).flatten()

        scalers[feature] = scaler

    dtrain = xgb.DMatrix(data=train_scaled, label=train['Close'])
    dtest = xgb.DMatrix(data=test_scaled, label=test['Close'])

    # Define the XGBoost parameters
    xgb_params = {'objective': 'reg:squarederror','booster': 'gbtree','eval_metric': 'rmse','colsample_bytree': 1,'eta': eta, 'max_depth': max_depth, 'min_child_weight': min_child_weight,  'subsample': 1.0}

    # Train the XGBoost model
    xgb_model = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=10000,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=5,
        verbose_eval=1000
    )

    
    # Calculate feature importance
    feature_importance = xgb_model.get_fscore()
    total_importance = sum(feature_importance.values())
    feature_importance = {k: v / total_importance for k, v in feature_importance.items()}  # Normalize the importance scores

    # Sort the features based on importance score
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    sorted_features_names, sorted_features_importance = zip(*sorted_features)
    
    # Map encoded feature names back to original names
    mapping_dict = {f'f{i}': col_name for i, col_name in enumerate(df2.drop(columns=['Close', 'MA_60', 'MA_250']).columns)}
    sorted_features_names_original = [mapping_dict[encoded_name] for encoded_name in sorted_features_names]
    
    print("\nFeature Importance:")
    for feature, importance in zip(sorted_features_names_original, sorted_features_importance):
        print(f"{feature}: {importance}")
    
    # Store feature importance in a dataframe
    feature_importance_df = pd.DataFrame({'Feature': sorted_features_names_original, 'Importance': sorted_features_importance})
        
    test_predicted = xgb_model.predict(dtest)
    rmsetest = np.sqrt(mean_squared_error(test_predicted, test['Close']))

    y_train_pred = xgb_model.predict(dtrain)
    rmsetrain = np.sqrt(mean_squared_error(y_train_pred,train['Close']))
    
    testdf = test.copy()
    testdf['Predicted Close'] = test_predicted
    testdf['Actual Close'] = testdf.pop('Close')
    testdf['MA_60'] = test_ma60
    testdf['MA_250'] = test_ma250

    testdf = testdf.reset_index()

    xgbtestviz,accuracy = output(testdf)
    get_viz(xgbtestviz,plotname,modelname)

    print(xgbtestviz[['Date','Previous Close - Actual','Predicted Price','Price Change']].tail(1))
    next_day_prediction = xgbtestviz['Predicted Price'].tail(1)
    prediction_2days = xgbtestviz.tail(2)
    
    return xgbtestviz, next_day_prediction,prediction_2days,rmsetest,rmsetrain,accuracy,feature_importance_df


