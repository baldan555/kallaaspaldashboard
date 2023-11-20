import random
import duckdb
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import statsmodels.api as sm
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import mysql.connector
from itertools import cycle
from datetime import date
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import mysql.connector
import math
import datetime as dt
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.layers import Conv1D, GRU, Dense
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.subplots as sp
import gspread
from oauth2client.service_account import ServiceAccountCredentials


st.set_page_config(page_icon="kalla aspal icon.png", page_title="Kalla Aspal Dash", layout="wide")



st.sidebar.image("logo kalla aspal.png", output_format="PNG", width=300)
page = st.sidebar.selectbox("Select a Page", ["Dashboard", "TrainPredictions", "Other Page"])

st.markdown(
        f"""
        <style>
        .centered-text {{
            text-align: center;
        }}
        </style>
        <div class="centered-text">
            <h1>Kalla Aspal | Price Analysis Dashboard</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
if page == "Dashboard":
    
    

    connection = mysql.connector.connect(
        host="217.21.73.32",
        user="u1801671_dmbsupredict",
        password="r.Sv[q!{=gl6",
        database="u1801671_dmbsupredict"
    )

    query = "SELECT * FROM harga_argus"
    cursor = connection.cursor(dictionary=True)
    cursor.execute(query)
    data = cursor.fetchall()
    df = pd.DataFrame(data)
    cursor.close()

    all_months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    with st.expander("Data Preview"):
        st.dataframe(
            df,
            column_config={"Year": st.column_config.NumberColumn(format="%d")},
        )

    df['Start_date'] = pd.to_datetime(df['Start_date'])
    with st.sidebar:
        st.title("Date Filter")
        start_date = st.date_input("Start Date", 
                                min_value=df['Start_date'].min().date(), 
                                max_value=df['Start_date'].max().date(),
                                value=pd.to_datetime('2021-01-01').date())
        end_date = st.date_input("End Date", 
                            min_value=df['Start_date'].min().date(), 
                            max_value=df['Start_date'].max().date(), 
                            value=df['Start_date'].max().date())

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    
    filtered_df = df[(df['Start_date'] >= start_date) & (df['Start_date'] <= end_date)]

    col1, col_spacer, col2 = st.columns([1, 0.1, 1])

    with col1:
        fig = px.line(filtered_df, x='Start_date', y=['Argus_High', 'Argus_Low', 'Argus_Mid'],
                    title="Argus Prices Over Time", labels={'variable': 'Price Type', 'value': 'Price'},
                    hover_data={'variable': True, 'Start_date': True})
        fig.update_layout(title_x=0.4)  
        st.markdown(
            f'<style> .css-1ksj3el {{ width: 90%; }} </style>',
            unsafe_allow_html=True
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_spacer:
        st.write("")

    with col2:
        st.dataframe(
            filtered_df.tail(10),  # Menggunakan fungsi tail(15) untuk menampilkan 15 baris terakhir
            column_config={"Year": st.column_config.NumberColumn(format="%d")},
        )

    


    credentials = {
        "type": "service_account",
        "project_id": "bsudashboard",
        "private_key_id": "f538312637b350162b50ea8eaf7b0221ba99db3f",
        "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDaYze1Uxbz1HT1\nApn8Q5YdBwFvF/xh6iJvksZsY60MrIUGL+UIzoV53ULlHnpyxhPd4TLZkN7jg5M8\niQEusC+747wO/0Y1oD+hmyYVpw5aZhKb2eaUTJKe+IzUh/PSYwNZFlmqF8rJiseI\naRbdFJCaHv2rlG+aFr5hq6DKFVC19kNFSnKwOGXL/yA+n3LnoIDAAVDwkhbf9qW+\nFS2//F0uDxQldA43gk/H3wmgKVL08fQTDU2vPX1mKLgjy01UvEDi1WWxH2QzH+Us\nn3sMO4A3aBQtHT4Sjg+pReUJmiClvZEPFw//qQlOVLKMQaOZmXJIZuyTNpr42vTq\nEmxe3Qs9AgMBAAECggEAJrGPKIevCKczTMj39xATjBOWb7RYgZG+JbYkBYbYnacA\nj8LTKg8La7bmGBnhGK5bTUQOIinOsBF7OC3PIZJSdtAoHSEiM7SvqYbHhe0aOC6f\nK/F87vgtP1DrBOoMq5FjAbkZzG8YPULjTRpPlH7FCyWkDO9JIcxXUY9a9ws1SHPk\ni/LueDGHDA9q2wH4N8q1qDMrQNJJIaXyw6C9yO3XBgMv8O3mKQ0XBsufXu9ZSMnB\n1IeYghNabEOSD3Zy9iwpfDXkKad2dZFPngdQVkRod7uC3M/bEulCQP+BGg7R2XZN\nUJvLg46kTO+ccGh3RnqeQrD6ggoq/n5ITHslDMoVSQKBgQD0kQyg9+/O4h7qdKxy\nZjI0TYDkIuu7ltkSl3tmw77LhakSgFVOKxRXgiRgALuAg6j5NR+VW0jUT++39cfk\nH766hBHoMqPifnl6dJKH+VSneMcWCgEbHLQbbFxwd1FdbuYgIf6QeDXja1Dv6Li2\n/ybincdV5AHyg2Mkv5SMj1lmZQKBgQDkmNwrxD5xWBwaBdTs5n6FQ3ECi1Urv78o\nEZAEn7rmTbi1uvjLgLyQeU3Qj7XdmFhkuRceBRCcMq3zcQpXPefMyji+FY3isvwv\nasMaDO1eBkOtodtRA/Ff/Z8SxWxR/6DA/CawH4Ynw7iwUrmay4O/LPlvbhIPhI7w\na39YXln3+QKBgCd2jFIhD8oTRKfSYc0n3mwAIY8ub1lasvZECT6A0XWnwKtil+FA\nJE0mi9YN6DOJiyeDQfBoxWy05VqKx1FxKeaQzw7ntPgkLlfrHYXhXp6TZ5SP0rPc\nQte1+YiQSEnKTknbEmefbhqfQ3Ui2UN2EoE4smSSoRnPx5yzaTXxH5zdAoGBAJq5\nM9yTuUfyQSMD63hlEiGU4jq8n+s6n2G5cWcPzervlCO854SmH/sRf6+4dXoFEV8r\nQBBbkUCHtOFfE4a1Kd2cygcug5h7WOGnO2K4eTj4uHofYVhWeNtlyHNmGMjvYpj8\nkhsHW0Ay4yDE/UdelE+jWzkmoSbLrAviVpEzW2XRAoGAKte0E+nIGq+ug6vj57Q3\n5rFRMZw4FVWCvTDIf0oxXaaK09jHbFzHMcwl7c4fJ7M3bY9dAc1bFme48eCBT8rP\nKxped/WLmuuqTkDRusSuiSWeDdg3+XQoWWet0mf1mjHm9j1wDHRrBtB+LR+o90u2\nitJyLI1RV+8/ewLu963xe5w=\n-----END PRIVATE KEY-----\n",
        "client_email": "bsudash@bsudashboard.iam.gserviceaccount.com",
        "client_id": "103544970533878376842",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/bsudash%40bsudashboard.iam.gserviceaccount.com",
        "universe_domain": "googleapis.com"
    }

    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    credentials_obj = ServiceAccountCredentials.from_json_keyfile_dict(credentials, scope)
    gc = gspread.authorize(credentials_obj)


    spreadsheet_name = "brentcrudeoil"  
    worksheet_name = "dailybrentoil"  

    try:
        sh = gc.open(spreadsheet_name)
        worksheet = sh.worksheet(worksheet_name)
    except gspread.SpreadsheetNotFound:
        st.error(f"Spreadsheet '{spreadsheet_name}' not found. Make sure the name is correct.")
        st.stop()
    except gspread.WorksheetNotFound:
        st.error(f"Worksheet '{worksheet_name}' not found in '{spreadsheet_name}'. Make sure the name is correct.")
        st.stop()

    data = worksheet.get_all_values()

    df = pd.DataFrame(data[1:], columns=data[0])

    

    df['Date'] = pd.to_datetime(df['Date'])
    df[['High', 'Low', 'Close']] = df[['High', 'Low', 'Close']].apply(pd.to_numeric)
    df['chg(high)'] = pd.to_numeric(df['chg(high)'], errors='coerce')
    df['chg(low)'] = pd.to_numeric(df['chg(low)'], errors='coerce')
    df['chg(close)'] = pd.to_numeric(df['chg(close)'], errors='coerce')

    

    import pandas as pd
    import streamlit as st
    import plotly.express as px
    from datetime import datetime, timedelta

    st.markdown("<br>", unsafe_allow_html=True)    
    st.markdown("<h1 style='text-align:center'>Brent Crude Oil Price</h1>", unsafe_allow_html=True)
    time_interval = st.selectbox("Select Time Interval", ["Daily", "Weekly", "Monthly", "Yearly", "Last 1 Month", "Last 2 Months", "Last 3 Months"], index = 6)

    if time_interval == "Daily":
        df_resampled = df.set_index('Date').resample('D').mean().reset_index()
    elif time_interval == "Weekly":
        df_resampled = df.set_index('Date').resample('W').mean().reset_index()
    elif time_interval == "Monthly":
        df_resampled = df.set_index('Date').resample('M').mean().reset_index()
    elif time_interval == "Yearly":
        df_resampled = df.set_index('Date').resample('Y').mean().reset_index()
    elif time_interval == "Last 1 Month":
        end_date = datetime.now()
        start_date = end_date - pd.DateOffset(months=1)
        df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        df_resampled = df_filtered.set_index('Date').resample('D').mean().reset_index()
    elif time_interval == "Last 2 Months":
        end_date = datetime.now()
        start_date = end_date - pd.DateOffset(months=2)
        df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        df_resampled = df_filtered.set_index('Date').resample('D').mean().reset_index()
    elif time_interval == "Last 3 Months":
        end_date = datetime.now()
        start_date = end_date - pd.DateOffset(months=3)
        df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        df_resampled = df_filtered.set_index('Date').resample('D').mean().reset_index()

    fig = px.line(df_resampled, x='Date', y=['High', 'Low'], title=f'Brent Crude Oil Price {time_interval}')
    fig.update_traces(connectgaps=True)
    fig.update_layout(title_x=0.4) 
    st.markdown(
        f'<style> .css-1ksj3el {{ width: 90%; }} </style>',
        unsafe_allow_html=True
    )
    st.plotly_chart(fig, use_container_width=True)


    
    df['chg(high)'] = df['chg(high)'].apply(lambda x: f'{float(x):+.2f}')
    df['chg(low)'] = df['chg(low)'].apply(lambda x: f'{float(x):+.2f}')
    df['chg(close)'] = df['chg(close)'].apply(lambda x: f'{float(x):+.2f}')

    def color_and_align(val):
        color = 'red' if float(val) < 0 else 'green'
        return f'color: {color}; text-align: right;'

    st.markdown("<h2 style='text-align:center'>Brent Crude Oil Futures Continues Price Board</h2>", unsafe_allow_html=True)

    df_style = df.tail(10).style.applymap(color_and_align, subset=['chg(high)', 'chg(low)', 'chg(close)'])

    st.markdown(
        f'<style> .css-1ksj3el {{ width: 90%; }} </style>',
        unsafe_allow_html=True
    )
    st.dataframe(df_style, use_container_width=True)
    

    db_connection = mysql.connector.connect(
        host="217.21.73.32",
        user="u1801671_dmbsupredict",
        password="r.Sv[q!{=gl6",
        database="u1801671_dmbsupredict"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)



    st.markdown("<h1 style='text-align:center'>FOB Bitumen Predicted Price</h1>", unsafe_allow_html=True)

    

    col1, col2 = st.columns(2)


    with col1:

        query_lstm_high = "SELECT * FROM lstm_predict_high"
        query_lstm_low = "SELECT * FROM lstm_predict_low"

        
        cursor_lstm_high = connection.cursor(dictionary=True)
        cursor_lstm_high.execute(query_lstm_high)
        data_lstm_high = cursor_lstm_high.fetchall()
        df_lstm_high = pd.read_sql(query_lstm_high, con=db_connection)

        cursor_lstm_low = connection.cursor(dictionary=True)
        cursor_lstm_low.execute(query_lstm_low)
        data_lstm_low = cursor_lstm_low.fetchall()
        df_lstm_low = pd.read_sql(query_lstm_low, con=db_connection)

        df_combined = pd.merge(df_lstm_low, df_lstm_high, on='Timestamp', suffixes=('_low', '_high'))
        df_combined['Predicted_price_mid'] = (df_combined['Predicted_price_low'] + df_combined['Predicted_price_high']) / 2

        df_combined.set_index('Timestamp', inplace=True)

    
        fig_combined = px.line(df_combined, y=['Predicted_price_low', 'Predicted_price_mid', 'Predicted_price_high'],
                            line_shape='linear',  
                            line_dash_sequence=['solid', 'dot', 'dash'],  
                            labels={'value': 'Predicted Price'},
                            color_discrete_sequence=['darkblue', 'lightblue', 'red'],
                            title='LSTM Model Predicted FOB Bitumen Price')

        fig_combined.add_trace(go.Scatter(x=df_combined.index[15:],
                                    y=df_combined['Predicted_price_low'].iloc[15:],
                                    mode='lines+markers',
                                    line=dict(dash='dot'),
                                    marker=dict(color='darkblue'),
                                    name='Predicted_price_low (dots)'))

        fig_combined.add_trace(go.Scatter(x=df_combined.index[15:],
                                    y=df_combined['Predicted_price_mid'].iloc[15:],
                                    mode='lines+markers',
                                    line=dict(dash='dot'),
                                    marker=dict(color='lightblue'),
                                    name='Predicted_price_low (dots)'))

        fig_combined.add_trace(go.Scatter(x=df_combined.index[15:],
                                    y=df_combined['Predicted_price_high'].iloc[15:],
                                    mode='lines+markers',
                                    line=dict(dash='dot'),
                                    marker=dict(color='red'),
                                    name='Predicted_price_low (dots)'))                            


        fig_combined.update_layout(title_x=0.4)  
        st.markdown(
            f'<style> .css-1ksj3el {{ width: 90%; }} </style>',
            unsafe_allow_html=True
        )
        st.plotly_chart(fig_combined, use_container_width=True)

    

        combined_predictions = pd.concat([df_combined['Predicted_price_high'], df_combined['Predicted_price_mid'], df_combined['Predicted_price_low']], axis=1)
        combined_predictions.columns = ['High', 'Mid', 'Low']

        st.markdown(
            f'<style> .css-1ksj3el {{ width: 90%; }} </style>',
            unsafe_allow_html=True
        )
        st.dataframe(combined_predictions.iloc[15:].style.set_properties(use_container_width=True))

    with col2:
        query_conv1d_high = "SELECT * FROM convdgru_predict_high"
        query_conv1d_low = "SELECT * FROM convdgru_predict_low"

        title_conv1d = 'CONV1DGRU Predicted Price'

        cursor_conv1d_high = connection.cursor(dictionary=True)
        cursor_conv1d_high.execute(query_conv1d_high)
        data_conv1d_high = cursor_conv1d_high.fetchall()
        df_conv1d_high = pd.read_sql(query_conv1d_high, con=db_connection)

        cursor_conv1d_low = connection.cursor(dictionary=True)
        cursor_conv1d_low.execute(query_conv1d_low)
        data_conv1d_low = cursor_conv1d_low.fetchall()
        df_conv1d_low = pd.read_sql(query_conv1d_low, con=db_connection)



        df_combined_conv1d = pd.merge(df_conv1d_low, df_conv1d_high, on='Timestamp', suffixes=('_low', '_high'))
        df_combined_conv1d['Predicted_price_mid'] = (df_combined_conv1d['Predicted_price_low'] + df_combined_conv1d['Predicted_price_high']) / 2

        df_combined_conv1d.set_index('Timestamp', inplace=True)
    
        fig_combined_conv1d = px.line(df_combined_conv1d, y=['Predicted_price_low', 'Predicted_price_mid', 'Predicted_price_high'],
                            line_shape='linear',  
                            line_dash_sequence=['solid', 'dot', 'dash'],  
                            labels={'value': 'Predicted Price'},
                            color_discrete_sequence=['darkblue', 'lightblue', 'red'],
                            title='Conv1dGru Model Predicted FOB Bitumen Price')

        fig_combined_conv1d.add_trace(go.Scatter(x=df_combined_conv1d.index[15:],
                                    y=df_combined_conv1d['Predicted_price_low'].iloc[15:],
                                    mode='lines+markers',
                                    line=dict(dash='dot'),
                                    marker=dict(color='darkblue'),
                                    name='Predicted_price_low (dots)'))

        fig_combined_conv1d.add_trace(go.Scatter(x=df_combined_conv1d.index[15:],
                                    y=df_combined_conv1d['Predicted_price_mid'].iloc[15:],
                                    mode='lines+markers',
                                    line=dict(dash='dot'),
                                    marker=dict(color='lightblue'),
                                    name='Predicted_price_low (dots)'))

        fig_combined_conv1d.add_trace(go.Scatter(x=df_combined_conv1d.index[15:],
                                    y=df_combined_conv1d['Predicted_price_high'].iloc[15:],
                                    mode='lines+markers',
                                    line=dict(dash='dot'),
                                    marker=dict(color='red'),
                                    name='Predicted_price_low (dots)'))   

        fig_combined_conv1d.update_layout(title_x=0.4)  
        st.markdown(
            f'<style> .css-1ksj3el {{ width: 90%; }} </style>',
            unsafe_allow_html=True
        )
        st.plotly_chart(fig_combined_conv1d, use_container_width=True)

        combined_predictions_conv1d = pd.concat([df_combined_conv1d['Predicted_price_high'], df_combined_conv1d['Predicted_price_mid'], df_combined_conv1d['Predicted_price_low']], axis=1)
        combined_predictions_conv1d.columns = ['High', 'Mid', 'Low']

        st.markdown(
            f'<style> .css-1ksj3el {{ width: 90%; }} </style>',
            unsafe_allow_html=True
        )
        st.dataframe(combined_predictions_conv1d.iloc[15:].style.set_properties(use_container_width=True))


    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align:center'>Model Score Evaluation</h1>", unsafe_allow_html=True)

    with st.expander("Detail", expanded=False):
        
        
        db_connection = mysql.connector.connect(
            host="217.21.73.32",
            user="u1801671_dmbsupredict",
            password="r.Sv[q!{=gl6",
            database="u1801671_dmbsupredict"
        )

        cursor = db_connection.cursor()

        cursor.execute("SELECT * FROM convdgru_r2score_high")
        conv_r2_high = cursor.fetchall()

        cursor.execute("SELECT * FROM convdgru_maescore_high")
        conv_mae_high = cursor.fetchall()

        cursor.execute("SELECT * FROM convdgru_msescore_high")
        conv_mse_high = cursor.fetchall()

        cursor.execute("SELECT * FROM convdgru_rmsescore_high")
        conv_rmse_high = cursor.fetchall()



        cursor.execute("SELECT * FROM convdgru_r2score_low")
        conv_r2_low = cursor.fetchall()

        cursor.execute("SELECT * FROM convdgru_maescore_low")
        conv_mae_low = cursor.fetchall()

        cursor.execute("SELECT * FROM convdgru_msescore_low")
        conv_mse_low = cursor.fetchall()

        cursor.execute("SELECT * FROM convdgru_rmsescore_low")
        conv_rmse_low = cursor.fetchall()

        

        cursor.execute("SELECT * FROM lstm_r2score_high")
        lstm_r2_high = cursor.fetchall()

        cursor.execute("SELECT * FROM lstm_maescore_high")
        lstm_mae_high = cursor.fetchall()

        cursor.execute("SELECT * FROM lstm_msescore_high")
        lstm_mse_high = cursor.fetchall()

        cursor.execute("SELECT * FROM lstm_rmsescore_high")
        lstm_rmse_high = cursor.fetchall()

      

        cursor.execute("SELECT * FROM lstm_r2score_low")
        lstm_r2_low = cursor.fetchall()

        cursor.execute("SELECT * FROM lstm_maescore_low")
        lstm_mae_low = cursor.fetchall()

        cursor.execute("SELECT * FROM lstm_msescore_low")
        lstm_mse_low = cursor.fetchall()

        cursor.execute("SELECT * FROM lstm_rmsescore_low")
        lstm_rmse_low = cursor.fetchall()

        df_conv_r2_high = pd.DataFrame(conv_r2_high, columns=['train', 'test'])
        df_conv_mae_high = pd.DataFrame(conv_mae_high, columns=['train', 'test'])
        df_conv_mse_high = pd.DataFrame(conv_mse_high, columns=['train', 'test'])
        df_conv_rmse_high = pd.DataFrame(conv_rmse_high, columns=['train', 'test'])


        df_conv_r2_low = pd.DataFrame(conv_r2_low, columns=['train', 'test'])
        df_conv_mae_low = pd.DataFrame(conv_mae_low, columns=['train', 'test'])
        df_conv_mse_low = pd.DataFrame(conv_mse_low, columns=['train', 'test'])
        df_conv_rmse_low = pd.DataFrame(conv_rmse_low, columns=['train', 'test'])

        df_lstm_r2_high = pd.DataFrame(lstm_r2_high, columns=['train', 'test'])
        df_lstm_mae_high = pd.DataFrame(lstm_mae_high, columns=['train', 'test'])
        df_lstm_mse_high = pd.DataFrame(lstm_mse_high, columns=['train', 'test'])
        df_lstm_rmse_high = pd.DataFrame(lstm_rmse_high, columns=['train', 'test'])


        df_lstm_r2_low = pd.DataFrame(lstm_r2_low, columns=['train', 'test'])
        df_lstm_mae_low = pd.DataFrame(lstm_mae_low, columns=['train', 'test'])
        df_lstm_mse_low = pd.DataFrame(lstm_mse_low, columns=['train', 'test'])
        df_lstm_rmse_low = pd.DataFrame(lstm_rmse_low, columns=['train', 'test'])


        
       
        
        fig = sp.make_subplots(rows=1, cols=4, subplot_titles=('R2 Scores', 'MAE Scores', 'MSE Scores', 'RMSE Scores'))
        fig.add_trace(go.Bar(
            x=['Training', 'Testing'],
            y=df_conv_r2_high.iloc[0],
            text=df_conv_r2_high.iloc[0],
            textposition='auto',
            marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
        ), row=1, col=1)

       
        fig.add_trace(go.Bar(
            x=['Training', 'Testing'],
            y=df_conv_mae_high.iloc[0],
            text=df_conv_mae_high.iloc[0],
            textposition='auto',
            marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
        ), row=1, col=2)

        fig.add_trace(go.Bar(
            x=['Training', 'Testing'],
            y=df_conv_mse_high.iloc[0],
            text=df_conv_mse_high.iloc[0],
            textposition='auto',
            marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
        ), row=1, col=3)

        fig.add_trace(go.Bar(
            x=['Training', 'Testing'],
            y=df_conv_rmse_high.iloc[0],
            text=df_conv_rmse_high.iloc[0],
            textposition='auto',
            marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
        ), row=1, col=4)

        fig.update_layout(height=400, width=1200, title_text="ConvDGRU Evaluation Scores for High Price", title_x=0.4)

        st.plotly_chart(fig)



        fig = sp.make_subplots(rows=1, cols=4, subplot_titles=('R2 Scores', 'MAE Scores', 'MSE Scores', 'RMSE Scores'))
        fig.add_trace(go.Bar(
            x=['Training', 'Testing'],
            y=df_conv_r2_low.iloc[0],
            text=df_conv_r2_low.iloc[0],
            textposition='auto',
            marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
        ), row=1, col=1)

       
        fig.add_trace(go.Bar(
            x=['Training', 'Testing'],
            y=df_conv_mae_low.iloc[0],
            text=df_conv_mae_low.iloc[0],
            textposition='auto',
            marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
        ), row=1, col=2)

        fig.add_trace(go.Bar(
            x=['Training', 'Testing'],
            y=df_conv_mse_low.iloc[0],
            text=df_conv_mse_low.iloc[0],
            textposition='auto',
            marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
        ), row=1, col=3)

        fig.add_trace(go.Bar(
            x=['Training', 'Testing'],
            y=df_conv_rmse_low.iloc[0],
            text=df_conv_rmse_low.iloc[0],
            textposition='auto',
            marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
        ), row=1, col=4)

        fig.update_layout(height=400, width=1200, title_text="ConvDGRU Evaluation Scores for Low Price", title_x=0.5)

        st.plotly_chart(fig)



        fig = sp.make_subplots(rows=1, cols=4, subplot_titles=('R2 Scores', 'MAE Scores', 'MSE Scores', 'RMSE Scores'))
        fig.add_trace(go.Bar(
            x=['Training', 'Testing'],
            y=df_lstm_r2_high.iloc[0],
            text=df_lstm_r2_high.iloc[0],
            textposition='auto',
            marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
        ), row=1, col=1)

       
        fig.add_trace(go.Bar(
            x=['Training', 'Testing'],
            y=df_lstm_mae_high.iloc[0],
            text=df_lstm_mae_high.iloc[0],
            textposition='auto',
            marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
        ), row=1, col=2)

        fig.add_trace(go.Bar(
            x=['Training', 'Testing'],
            y=df_lstm_mse_high.iloc[0],
            text=df_lstm_mse_high.iloc[0],
            textposition='auto',
            marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
        ), row=1, col=3)

        fig.add_trace(go.Bar(
            x=['Training', 'Testing'],
            y=df_lstm_rmse_high.iloc[0],
            text=df_lstm_rmse_high.iloc[0],
            textposition='auto',
            marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
        ), row=1, col=4)

        fig.update_layout(height=400, width=1200, title_text="LSTM Evaluation Scores for High Price", title_x=0.5)

        st.plotly_chart(fig)







        fig = sp.make_subplots(rows=1, cols=4, subplot_titles=('R2 Scores', 'MAE Scores', 'MSE Scores', 'RMSE Scores'))
        fig.add_trace(go.Bar(
            x=['Training', 'Testing'],
            y=df_lstm_r2_low.iloc[0],
            text=df_lstm_r2_low.iloc[0],
            textposition='auto',
            marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
        ), row=1, col=1)

       
        fig.add_trace(go.Bar(
            x=['Training', 'Testing'],
            y=df_lstm_mae_low.iloc[0],
            text=df_lstm_mae_low.iloc[0],
            textposition='auto',
            marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
        ), row=1, col=2)

        fig.add_trace(go.Bar(
            x=['Training', 'Testing'],
            y=df_lstm_mse_low.iloc[0],
            text=df_lstm_mse_low.iloc[0],
            textposition='auto',
            marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
        ), row=1, col=3)

        fig.add_trace(go.Bar(
            x=['Training', 'Testing'],
            y=df_lstm_rmse_low.iloc[0],
            text=df_lstm_rmse_low.iloc[0],
            textposition='auto',
            marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
        ), row=1, col=4)

        fig.update_layout(height=400, width=1200, title_text="LSTM Evaluation Scores for Low Price", title_x=0.5)

        st.plotly_chart(fig)


        cursor.close()
        db_connection.close()




if page == "TrainPredictions":
    connection = mysql.connector.connect(
        host="217.21.73.32",
        user="u1801671_dmbsupredict",
        password="r.Sv[q!{=gl6",
        database="u1801671_dmbsupredict"
    )

    query = "SELECT * FROM harga_argus"
    cursor = connection.cursor(dictionary=True)
    cursor.execute(query)
    data = cursor.fetchall()
    df = pd.DataFrame(data)
    cursor.close()

    all_months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    with st.expander("Data Preview"):
        st.dataframe(
            df,
            column_config={"Year": st.column_config.NumberColumn(format="%d")},
        )

    df['Start_date'] = pd.to_datetime(df['Start_date'])
    with st.sidebar:
        st.title("Date Filter")
        start_date = st.date_input("Start Date", min_value=df['Start_date'].min().date(), max_value=df['Start_date'].max().date(), value=df['Start_date'].min().date())
        end_date = st.date_input("End Date", min_value=df['Start_date'].min().date(), max_value=df['Start_date'].max().date(), value=df['Start_date'].max().date())

        
        
        st.title("Insert New Data")
        max_date = date(2070, 12, 31)
        argus_high = st.number_input("Argus High Price", min_value=0.0, value=0.0, key="argus_high")
        argus_low = st.number_input("Argus Low Price", min_value=0.0, value=0.0, key="argus_low")
        argus_mid = st.number_input("Argus Mid Price", min_value=0.0, value=0.0, key="argus_mid")
        start_date_input = st.date_input("Start Date", min_value=df['Start_date'].min().date(),
                              max_value=max_date, value=df['Start_date'].min().date(),
                              key="start_date_input")
        end_date_input = st.date_input("End Date", min_value=df['Start_date'].min().date(),
                              max_value=max_date, value=df['Start_date'].max().date(),
                              key="end_date_input")

        if st.button("Insert Data"):
            connection = mysql.connector.connect(
                host="217.21.73.32",
                user="u1801671_dmbsupredict",
                password="r.Sv[q!{=gl6",
                database="u1801671_dmbsupredict"
            )

            cursor = connection.cursor()
            insert_query = "INSERT INTO harga_argus (Argus_High, Argus_Low, Argus_Mid, Start_date, End_date) " \
                           "VALUES (%s, %s, %s, %s, %s)"
            cursor.execute(insert_query, (argus_high, argus_low, argus_mid, start_date_input, end_date_input))
            connection.commit()
            cursor.close()
            st.success("Data inserted successfully.")

        st.title("Delete Data")
        selected_rows = st.multiselect("Select Rows to Delete", df['Start_date'])

        if st.button("Delete Selected Rows"):
            connection = mysql.connector.connect(
                host="217.21.73.32",
                user="u1801671_dmbsupredict",
                password="r.Sv[q!{=gl6",
                database="u1801671_dmbsupredict"
            )
            cursor = connection.cursor()

 
            for start_date in selected_rows:
                delete_query = "DELETE FROM harga_argus WHERE Start_date = %s"
                cursor.execute(delete_query, (start_date,))
                connection.commit()

            cursor.close()
            connection.close()
            st.success("Selected rows have been deleted.")

        
        df = df[~df['Start_date'].isin(selected_rows)]

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    filtered_df = df[(df['Start_date'] >= start_date) & (df['Start_date'] <= end_date)]

    fig = px.line(filtered_df, x='Start_date', y=['Argus_High', 'Argus_Low', 'Argus_Mid'],
                title="Argus Prices Over Time", labels={'variable': 'Price Type', 'value': 'Price'},
                hover_data={'variable': False, 'Start_date': False})

    st.markdown(
        f'<style> .css-1ksj3el {{ width: 90%; }} </style>',
        unsafe_allow_html=True
    )
    st.plotly_chart(fig, use_container_width=True)


    st.markdown("<h1 style='text-align:center'>Train Predict Model</h1>", unsafe_allow_html=True)
    model_options = ['Conv1dGru', 'Support Vector Machine (SVM)', 'LSTM']
    selected_data = st.selectbox("Select target column for training", ['Argus_High', 'Argus_Low'])
    selected_model = st.selectbox("Select the model", model_options)
    pred_week = st.slider("Select the number of Weeks to predict", min_value=1, max_value=30, value=10)
    if selected_model == 'Conv1dGru':
        if selected_data == 'Argus_High':
            if st.button("Train"):
                if connection.is_connected():
                    cursor = connection.cursor(dictionary=True)
                    cursor.execute("SELECT * FROM harga_argus")
                    data = cursor.fetchall()
                    df = pd.DataFrame(data)
                    cursor.close()
                    connection.close()

                    with st.spinner("In Progress..."):
                    
                        trans = ['Start_date', 'End_date']

                        for column in trans:
                            df[column] = df[column].astype(str)
                            df[column] = df[column].str.replace(r'\s+', '', regex=True)
                            df[column] = df[column].str[:4] + '-' + df[column].str[4:7] + df[column].str[7:]

                        df['Start_date'] = pd.to_datetime(df['Start_date'])
                        df['End_date'] = pd.to_datetime(df['End_date'])

                        argus = df[['Start_date', 'Argus_High']]

                        copy_price = argus.copy()
                        del argus['Start_date']
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        argus = scaler.fit_transform(np.array(argus).reshape(-1, 1))

                        training_size = int(len(argus) * 0.65)
                        test_size = len(argus) - training_size
                        train_data, test_data = argus[0:training_size, :], argus[training_size:len(argus), :1]

                        def create_dataset(dataset, time_step=0):
                            dataX, dataY = [], []
                            for i in range(len(dataset) - time_step - 1):
                                a = dataset[i:(i + time_step), 0]
                                dataX.append(a)
                                dataY.append(dataset[i + time_step, 0])
                            return np.array(dataX), np.array(dataY)

                        time_step = 15
                        X_train, y_train = create_dataset(train_data, time_step)
                        X_test, y_test = create_dataset(test_data, time_step)

                        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

                        model = Sequential()
                        model.add(Conv1D(filters=38, kernel_size=9, activation='relu', input_shape=(time_step, 1)))
                        model.add(GRU(128, return_sequences=True))
                        model.add(GRU(128))
                        model.add(Dense(1, activation='relu'))

                        learning_rate = 0.0005988691849238099
                        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate))

                        batch_size = 32
                        num_epochs = 79
                        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=num_epochs, batch_size=batch_size, verbose=1)

                        loss = model.evaluate(X_test, y_test)

                        train_predict = model.predict(X_train)
                        test_predict = model.predict(X_test)

                        train_predict = scaler.inverse_transform(train_predict)
                        test_predict = scaler.inverse_transform(test_predict)
                        original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
                        original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))


                        look_back = time_step

                        trainPredictPlot = np.empty_like(argus)
                        trainPredictPlot[:, :] = np.nan
                        trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

                        testPredictPlot = np.empty_like(argus)
                        testPredictPlot[:, :] = np.nan
                        testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(argus) - 1, :] = test_predict

                        plotdf = pd.DataFrame({
                            'Start_date': copy_price['Start_date'],
                            'original_price': copy_price['Argus_High'],
                            'train_predicted': trainPredictPlot.reshape(1, -1)[0].tolist(),
                            'test_predicted': testPredictPlot.reshape(1, -1)[0].tolist()
                        })

                        fig = px.line(plotdf, x=plotdf['Start_date'],
                                    y=[plotdf['original_price'], plotdf['train_predicted'], plotdf['test_predicted']],
                                    labels={'value': 'price', '': 'Date'})
                        fig.update_layout(title_text='',
                                        plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='')

                        names = cycle(['Harga Aktual', 'Train predicted price', 'Test predicted price'])
                        fig.for_each_trace(lambda t: t.update(name=next(names), line_width=4,))

                        fig.update_xaxes(showgrid=False)
                        fig.update_yaxes(showgrid=False)

                        st.markdown(
                            f'<style> .css-1ksj3el {{ width: 90%; }} </style>',
                            unsafe_allow_html=True
                        )
                        st.markdown("<h1 style='text-align:center'>Train Test Plot</h1>", unsafe_allow_html=True)
                        st.plotly_chart(fig, use_container_width=True)

                    
                        st.markdown("<h1 style='text-align:center'>Model Evaluation Metrics</h1>", unsafe_allow_html=True)

                        train_rmse = math.sqrt(mean_squared_error(original_ytrain, train_predict))
                        train_mse = mean_squared_error(original_ytrain, train_predict)
                        train_mae = mean_absolute_error(original_ytrain, train_predict)
                        train_r2 = r2_score(original_ytrain, train_predict)

                        test_rmse = math.sqrt(mean_squared_error(original_ytest, test_predict))
                        test_mse = mean_squared_error(original_ytest, test_predict)
                        test_mae = mean_absolute_error(original_ytest, test_predict)
                        test_r2 = r2_score(original_ytest, test_predict)
                        

                        db_config = {
                            'host': '217.21.73.32',
                            'user': 'u1801671_dmbsupredict',
                            'password': 'r.Sv[q!{=gl6',
                            'database': 'u1801671_dmbsupredict'
                        }

                        db_connection = mysql.connector.connect(**db_config)
                        cursor = db_connection.cursor()

                        try:
                            cursor.execute("DELETE FROM convdgru_r2score_high")
                            cursor.execute("DELETE FROM convdgru_maescore_high")
                            cursor.execute("DELETE FROM convdgru_msescore_high")
                            cursor.execute("DELETE FROM convdgru_rmsescore_high")

                            cursor.execute("INSERT INTO convdgru_r2score_high (train_conv1dgru, test_conv1dgru) VALUES (%s, %s)", (train_r2, test_r2))
                            cursor.execute("INSERT INTO convdgru_maescore_high (train, test) VALUES (%s, %s)", (train_mae, test_mae))
                            cursor.execute("INSERT INTO convdgru_msescore_high (train, test) VALUES (%s, %s)", (train_mse, test_mse))
                            cursor.execute("INSERT INTO convdgru_rmsescore_high (train, test) VALUES (%s, %s)", (train_rmse, test_rmse))

                            db_connection.commit()

                        except mysql.connector.Error as err:
                            print(f"Error: {err}")
                            db_connection.rollback()

                        finally:
                            cursor.close()
                            db_connection.close()

                        
                        coll1, coll2 = st.columns(2)
                        
                        with coll1:
                            fig = go.Figure()

                            fig.add_trace(go.Bar(
                                x=['Training R-squared', 'Testing R-squared'],
                                y=[train_r2, test_r2],
                                text=[round(train_r2, 2), round(test_r2, 2)],
                                textposition='auto',
                                marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
                            ))

                            fig.update_layout(
                                title='R-squared Score',
                                xaxis_title='',
                                yaxis_title='',
                                title_x=0.5, 
                            )

                            st.plotly_chart(fig)

                        with coll2:
                            fig = go.Figure()

                            fig.add_trace(go.Bar(
                                x=['Training MAE', 'Testing MAE'],
                                y=[train_mae, test_mae],
                                text=[round(train_mae, 2), round(test_mae, 2)],
                                textposition='auto',
                                marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
                            ))

                            fig.update_layout(
                                title='MAE Score',
                                xaxis_title='',
                                yaxis_title='',
                                title_x=0.5, 
                            
                            )

                            st.plotly_chart(fig)

                        colll1, colll2 = st.columns(2)
                        
                        with colll1:
                            fig = go.Figure()

                            fig.add_trace(go.Bar(
                                x=['Training MSE', 'Testing MSE'],
                                y=[train_mse, test_mse],
                                text=[round(train_mse, 2), round(test_mse, 2)],
                                textposition='auto',
                                marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
                            ))

                            fig.update_layout(
                                title='MSE Score',
                                xaxis_title='',
                                yaxis_title='',
                                title_x=0.5,  
                            
                            )

                            st.plotly_chart(fig)

                        with colll2:
                            fig = go.Figure()

                            fig.add_trace(go.Bar(
                                x=['Training RMSE', 'Testing RMSE'],
                                y=[train_rmse, test_rmse],
                                text=[round(train_rmse, 2), round(test_rmse, 2)],
                                textposition='auto',
                                marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
                            ))

                            fig.update_layout(
                                title='RMSE Score',
                                xaxis_title='',
                                yaxis_title='',
                                title_x=0.5,  
                                
                            )

                            st.plotly_chart(fig)



                        st.write('Col 2:')
                        st.subheader("Train Metrics:")
                        st.write(f"RMSE: {train_rmse:.2f}")
                        st.write(f"MSE: {train_mse:.2f}")
                        st.write(f"MAE: {train_mae:.2f}")
                        st.write(f"R-squared: {train_r2:.2f}")

                        st.subheader("Test Metrics:")
                        st.write(f"RMSE: {test_rmse:.2f}")
                        st.write(f"MSE: {test_mse:.2f}")
                        st.write(f"MAE: {test_mae:.2f}")
                        st.write(f"R-squared: {test_r2:.2f}")

            
                        st.subheader("Prediction Results")
                        st.write("Test Loss:", loss)

                        
                    



                        x_input = test_data[len(test_data) - time_step:].reshape(1, -1)
                        temp_input = list(x_input)
                        temp_input = temp_input[0].tolist()

                        lst_output = []
                        n_steps = time_step
                        i = 0
                        pred_week = pred_week

                        
                        while i < pred_week:
                            if len(temp_input) > time_step:
                                x_input = np.array(temp_input[1:])
                                x_input = x_input.reshape(1, -1)
                                x_input = x_input.reshape((1, n_steps, 1))

                                yhat = model.predict(x_input, verbose=0)
                                temp_input.extend(yhat[0].tolist())
                                temp_input = temp_input[1:]
                                lst_output.extend(yhat.tolist())

                                i = i + 1
                            else:
                                x_input = x_input.reshape((1, n_steps, 1))
                                yhat = model.predict(x_input, verbose=0)
                                temp_input.extend(yhat[0].tolist())
                                lst_output.extend(yhat.tolist())

                                i = i + 1

                        
                        start_date = pd.to_datetime(start_date)
                        end_date = pd.to_datetime(end_date)

                        Start_date = df['Start_date'].sort_values(ascending=False).iloc[16].strftime('%Y-%m-%d')
                        last_week=np.arange(1,time_step+1)
                        day_pred=np.arange(time_step+1,time_step+pred_week+1)
                        print(last_week)
                        print(day_pred)

                        temp_mat = np.empty((len(last_week)+pred_week+1,1))
                        temp_mat[:] = np.nan
                        temp_mat = temp_mat.reshape(1,-1).tolist()[0]
                        

                        last_original_week_value = temp_mat
                        next_predicted_week_value = temp_mat

                        last_original_week_value[0:time_step+1] = scaler.inverse_transform(argus[len(argus)-time_step:]).reshape(1,-1).tolist()[0]
                        
                        next_predicted_week_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]
                        
                        conv1dgru_results = {
                            'last_original_week_value': last_original_week_value,
                            'next_predicted_week_value': next_predicted_week_value,
                        }

                        new_pred_plot = pd.DataFrame({
                            'last_original_week_value':last_original_week_value,
                            'next_predicted_week_value':next_predicted_week_value,
                            
                        })

                        names = cycle(['Last 15 week close price','Predicted next 10 week price'])
                        new_pred_plot['Timestamp'] = pd.date_range(start=Start_date, periods=len(last_week)+pred_week+1, freq='w')


                        st.markdown("<h1 style='text-align:center'>Plot Prediction</h1>", unsafe_allow_html=True)

                        fig = px.line(new_pred_plot, x='Timestamp', y=['last_original_week_value', 'next_predicted_week_value'],
                                    labels={'value': 'Stock price'},
                                    title='Plot Prediction')

                        fig.update_layout(plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')

                        st.markdown(
                            f'<style> .css-1ksj3el {{ width: 90%; }} </style>',
                            unsafe_allow_html=True
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        
                        timestamps = pd.date_range(start=Start_date, periods=len(last_week) + pred_week + 1, freq='w')
                        prediction_results = pd.DataFrame({
                            'Timestamp': timestamps,
                            'Predicted next 10 week price': next_predicted_week_value
                        })

                        if 'last_week' in prediction_results:
                            prediction_results.drop('last_week', axis=1, inplace=True)

                        prediction_results.dropna(subset=['Predicted next 10 week price'], inplace=True)
                        st.write(prediction_results)

            
                        db_connection = mysql.connector.connect(
                            host="217.21.73.32",
                            user="u1801671_dmbsupredict",
                            password="r.Sv[q!{=gl6",
                            database="u1801671_dmbsupredict"
                        )

                        cursor = db_connection.cursor()
                        cursor.execute("TRUNCATE TABLE convdgru_predict_high")
                        for index, row in prediction_results.iterrows():
                            cursor.execute("INSERT INTO convdgru_predict_high (Timestamp, `Predicted_price`) VALUES (%s, %s)",
                                        (row['Timestamp'], row['Predicted next 10 week price']))

                        db_connection.commit()

                        cursor.close()
                        db_connection.close()


                else:
                    st.error("Failed to connect to the database. Check your database connection settings.")


        elif selected_data == 'Argus_Low':
            if st.button("Train"):
                if connection.is_connected():
                    cursor = connection.cursor(dictionary=True)
                    cursor.execute("SELECT * FROM harga_argus")
                    data = cursor.fetchall()
                    df = pd.DataFrame(data)
                    cursor.close()
                    connection.close()

                    with st.spinner("In Progress..."):
                        trans = ['Start_date', 'End_date']

                        for column in trans:
                            df[column] = df[column].astype(str)
                            df[column] = df[column].str.replace(r'\s+', '', regex=True)
                            df[column] = df[column].str[:4] + '-' + df[column].str[4:7] + df[column].str[7:]

                        df['Start_date'] = pd.to_datetime(df['Start_date'])
                        df['End_date'] = pd.to_datetime(df['End_date'])

                        argus = df[['Start_date', 'Argus_Low']]

                        copy_price = argus.copy()
                        del argus['Start_date']
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        argus = scaler.fit_transform(np.array(argus).reshape(-1, 1))

                        training_size = int(len(argus) * 0.65)
                        test_size = len(argus) - training_size
                        train_data, test_data = argus[0:training_size, :], argus[training_size:len(argus), :1]

                        def create_dataset(dataset, time_step=1):
                            dataX, dataY = [], []
                            for i in range(len(dataset) - time_step - 1):
                                a = dataset[i:(i + time_step), 0]
                                dataX.append(a)
                                dataY.append(dataset[i + time_step, 0])
                            return np.array(dataX), np.array(dataY)

                        time_step = 15
                        X_train, y_train = create_dataset(train_data, time_step)
                        X_test, y_test = create_dataset(test_data, time_step)

                        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        
                        model = Sequential()
                        model.add(Conv1D(filters=38, kernel_size=9, activation='relu', input_shape=(time_step, 1)))
                        model.add(GRU(128, return_sequences=True))
                        model.add(GRU(128))
                        model.add(Dense(1, activation='relu'))

            
                        learning_rate = 0.0005988691849238099
                        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate))

                
                        batch_size = 32
                        num_epochs = 79
                        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=num_epochs, batch_size=batch_size, verbose=1)

                        loss = model.evaluate(X_test, y_test)

                        train_predict = model.predict(X_train)
                        test_predict = model.predict(X_test)

                        train_predict = scaler.inverse_transform(train_predict)
                        test_predict = scaler.inverse_transform(test_predict)
                        original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
                        original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))


                        look_back = time_step

            
                        trainPredictPlot = np.empty_like(argus)
                        trainPredictPlot[:, :] = np.nan
                        trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

                        testPredictPlot = np.empty_like(argus)
                        testPredictPlot[:, :] = np.nan
                        testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(argus) - 1, :] = test_predict

                        plotdf = pd.DataFrame({
                            'Start_date': copy_price['Start_date'],
                            'original_price': copy_price['Argus_Low'],
                            'train_predicted': trainPredictPlot.reshape(1, -1)[0].tolist(),
                            'test_predicted': testPredictPlot.reshape(1, -1)[0].tolist()
                        })

            
                        fig = px.line(plotdf, x=plotdf['Start_date'],
                                    y=[plotdf['original_price'], plotdf['train_predicted'], plotdf['test_predicted']],
                                    labels={'value': 'price', '': 'Date'})
                        fig.update_layout(title_text='',
                                        plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='')

        
                        names = cycle(['Harga Aktual', 'Train predicted price', 'Test predicted price'])
                        fig.for_each_trace(lambda t: t.update(name=next(names), line_width=4,))

                
                        fig.update_xaxes(showgrid=False)
                        fig.update_yaxes(showgrid=False)

                        st.markdown(
                            f'<style> .css-1ksj3el {{ width: 90%; }} </style>',
                            unsafe_allow_html=True
                        )

                        st.markdown("<h1 style='text-align:center'>Train Test Plot</h1>", unsafe_allow_html=True)
                        st.plotly_chart(fig, use_container_width=True)

                    
                        st.markdown("<h1 style='text-align:center'>Model Evaluation Metrics</h1>", unsafe_allow_html=True)

                    
                        train_rmse = math.sqrt(mean_squared_error(original_ytrain, train_predict))
                        train_mse = mean_squared_error(original_ytrain, train_predict)
                        train_mae = mean_absolute_error(original_ytrain, train_predict)
                        train_r2 = r2_score(original_ytrain, train_predict)

                        test_rmse = math.sqrt(mean_squared_error(original_ytest, test_predict))
                        test_mse = mean_squared_error(original_ytest, test_predict)
                        test_mae = mean_absolute_error(original_ytest, test_predict)
                        test_r2 = r2_score(original_ytest, test_predict)
                        

                    
                        db_config = {
                            'host': '217.21.73.32',
                            'user': 'u1801671_dmbsupredict',
                            'password': 'r.Sv[q!{=gl6',
                            'database': 'u1801671_dmbsupredict'
                        }

                        db_connection = mysql.connector.connect(**db_config)
                        cursor = db_connection.cursor()

                        try:
                        
                            cursor.execute("DELETE FROM convdgru_r2score_low")
                            cursor.execute("DELETE FROM convdgru_maescore_low")
                            cursor.execute("DELETE FROM convdgru_msescore_low")
                            cursor.execute("DELETE FROM convdgru_rmsescore_low")

                        
                            cursor.execute("INSERT INTO convdgru_r2score_low (train, test) VALUES (%s, %s)", (train_r2, test_r2))
                            cursor.execute("INSERT INTO convdgru_maescore_low (train, test) VALUES (%s, %s)", (train_mae, test_mae))
                            cursor.execute("INSERT INTO convdgru_msescore_low (train, test) VALUES (%s, %s)", (train_mse, test_mse))
                            cursor.execute("INSERT INTO convdgru_rmsescore_low (train, test) VALUES (%s, %s)", (train_rmse, test_rmse))

                            db_connection.commit()

                        except mysql.connector.Error as err:
                            print(f"Error: {err}")
                            db_connection.rollback()

                        finally:
                            cursor.close()
                            db_connection.close()

                        
                        coll1, coll2 = st.columns(2)
                        
                        with coll1:
                            fig = go.Figure()

                            fig.add_trace(go.Bar(
                                x=['Training R-squared', 'Testing R-squared'],
                                y=[train_r2, test_r2],
                                text=[round(train_r2, 2), round(test_r2, 2)],
                                textposition='auto',
                                marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
                            ))

                            fig.update_layout(
                                title='R-squared Score',
                                xaxis_title='',
                                yaxis_title='',
                                title_x=0.5,  
                            )

                            st.plotly_chart(fig)

                        with coll2:
                            fig = go.Figure()

                            fig.add_trace(go.Bar(
                                x=['Training MAE', 'Testing MAE'],
                                y=[train_mae, test_mae],
                                text=[round(train_mae, 2), round(test_mae, 2)],
                                textposition='auto',
                                marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
                            ))

                            fig.update_layout(
                                title='MAE Score',
                                xaxis_title='',
                                yaxis_title='',
                                title_x=0.5, 
                            
                            )

                            st.plotly_chart(fig)

                        colll1, colll2 = st.columns(2)
                        
                        with colll1:
                            fig = go.Figure()

                            fig.add_trace(go.Bar(
                                x=['Training MSE', 'Testing MSE'],
                                y=[train_mse, test_mse],
                                text=[round(train_mse, 2), round(test_mse, 2)],
                                textposition='auto',
                                marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
                            ))

                            fig.update_layout(
                                title='MSE Score',
                                xaxis_title='',
                                yaxis_title='',
                                title_x=0.5,
                            
                            )

                            st.plotly_chart(fig)

                        with colll2:
                            fig = go.Figure()

                            fig.add_trace(go.Bar(
                                x=['Training RMSE', 'Testing RMSE'],
                                y=[train_rmse, test_rmse],
                                text=[round(train_rmse, 2), round(test_rmse, 2)],
                                textposition='auto',
                                marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
                            ))

                            fig.update_layout(
                                title='RMSE Score',
                                xaxis_title='',
                                yaxis_title='',
                                title_x=0.5,  
                                
                            )

                            st.plotly_chart(fig)



                        st.write('Col 2:')
                        st.subheader("Train Metrics:")
                        st.write(f"RMSE: {train_rmse:.2f}")
                        st.write(f"MSE: {train_mse:.2f}")
                        st.write(f"MAE: {train_mae:.2f}")
                        st.write(f"R-squared: {train_r2:.2f}")

                        st.subheader("Test Metrics:")
                        st.write(f"RMSE: {test_rmse:.2f}")
                        st.write(f"MSE: {test_mse:.2f}")
                        st.write(f"MAE: {test_mae:.2f}")
                        st.write(f"R-squared: {test_r2:.2f}")

            
                        st.subheader("Prediction Results")
                        st.write("Test Loss:", loss)



                        x_input = test_data[len(test_data) - time_step:].reshape(1, -1)
                        temp_input = list(x_input)
                        temp_input = temp_input[0].tolist()

                        lst_output = []
                        n_steps = time_step
                        i = 0
                        pred_week = pred_week

                        
                        while i < pred_week:
                            if len(temp_input) > time_step:
                                x_input = np.array(temp_input[1:])
                                x_input = x_input.reshape(1, -1)
                                x_input = x_input.reshape((1, n_steps, 1))

                                yhat = model.predict(x_input, verbose=0)
                                temp_input.extend(yhat[0].tolist())
                                temp_input = temp_input[1:]
                                lst_output.extend(yhat.tolist())

                                i = i + 1
                            else:
                                x_input = x_input.reshape((1, n_steps, 1))
                                yhat = model.predict(x_input, verbose=0)
                                temp_input.extend(yhat[0].tolist())
                                lst_output.extend(yhat.tolist())

                                i = i + 1


                        start_date = pd.to_datetime(start_date)
                        end_date = pd.to_datetime(end_date)

                        Start_date = df['Start_date'].sort_values(ascending=False).iloc[16].strftime('%Y-%m-%d')

                        last_week=np.arange(1,time_step+1)
                        day_pred=np.arange(time_step+1,time_step+pred_week+1)
                        print(last_week)
                        print(day_pred)

                        temp_mat = np.empty((len(last_week)+pred_week+1,1))
                        temp_mat[:] = np.nan
                        temp_mat = temp_mat.reshape(1,-1).tolist()[0]
                        

                        last_original_week_value = temp_mat
                        next_predicted_week_value = temp_mat

                        last_original_week_value[0:time_step+1] = scaler.inverse_transform(argus[len(argus)-time_step:]).reshape(1,-1).tolist()[0]
                        
                        next_predicted_week_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]
                        
                        conv1dgru_results = {
                            'last_original_week_value': last_original_week_value,
                            'next_predicted_week_value': next_predicted_week_value,
                        }

                        new_pred_plot = pd.DataFrame({
                            'last_original_week_value':last_original_week_value,
                            'next_predicted_week_value':next_predicted_week_value,
                            
                        })

                        names = cycle(['Last 15 week close price','Predicted next 10 week price'])
                        new_pred_plot['Timestamp'] = pd.date_range(start=Start_date, periods=len(last_week)+pred_week+1, freq='w')


                        st.markdown("<h1 style='text-align:center'>Plot Prediction</h1>", unsafe_allow_html=True)

                        fig = px.line(new_pred_plot, x='Timestamp', y=['last_original_week_value', 'next_predicted_week_value'],
                                    labels={'value': 'Stock price'},
                                    title='Plot Prediction')

                        fig.update_layout(plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')

                        st.markdown(
                            f'<style> .css-1ksj3el {{ width: 90%; }} </style>',
                            unsafe_allow_html=True
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        
                        timestamps = pd.date_range(start=Start_date, periods=len(last_week) + pred_week + 1, freq='w')
                        prediction_results = pd.DataFrame({
                            'Timestamp': timestamps,
                            'Predicted next 10 week price': next_predicted_week_value
                        })

                        if 'last_week' in prediction_results:
                            prediction_results.drop('last_week', axis=1, inplace=True)

                        prediction_results.dropna(subset=['Predicted next 10 week price'], inplace=True)
                        st.write(prediction_results)

                        
                        db_connection = mysql.connector.connect(
                            host="217.21.73.32",
                            user="u1801671_dmbsupredict",
                            password="r.Sv[q!{=gl6",
                            database="u1801671_dmbsupredict"
                        )

                        cursor = db_connection.cursor()
                        cursor.execute("TRUNCATE TABLE convdgru_predict_low")
                        for index, row in prediction_results.iterrows():
                            cursor.execute("INSERT INTO convdgru_predict_low (Timestamp, `Predicted_price`) VALUES (%s, %s)",
                                        (row['Timestamp'], row['Predicted next 10 week price']))

            
                        db_connection.commit()

                        cursor.close()
                        db_connection.close()


                else:
                    st.error("Failed to connect to the database. Check your database connection settings.")






#LSTM MODEL


    elif selected_model == 'LSTM':
        if selected_data == 'Argus_High':
            if st.button("Train"):
                if connection.is_connected():
                    cursor = connection.cursor(dictionary=True)
                    cursor.execute("SELECT * FROM harga_argus")
                    data = cursor.fetchall()
                    df = pd.DataFrame(data)
                    cursor.close()
                    connection.close()

                    with st.spinner("In Progress..."):
                        trans = ['Start_date', 'End_date']

                        for column in trans:
                            df[column] = df[column].astype(str)
                            df[column] = df[column].str.replace(r'\s+', '', regex=True)
                            df[column] = df[column].str[:4] + '-' + df[column].str[4:7] + df[column].str[7:]

                        df['Start_date'] = pd.to_datetime(df['Start_date'])
                        df['End_date'] = pd.to_datetime(df['End_date'])

                        argus = df[['Start_date', 'Argus_High']]

                        copy_price = argus.copy()
                        del argus['Start_date']
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        argus = scaler.fit_transform(np.array(argus).reshape(-1, 1))

                        training_size = int(len(argus) * 0.65)
                        test_size = len(argus) - training_size
                        train_data, test_data = argus[0:training_size, :], argus[training_size:len(argus), :1]

                        def create_dataset(dataset, time_step=1):
                            dataX, dataY = [], []
                            for i in range(len(dataset) - time_step - 1):
                                a = dataset[i:(i + time_step), 0]
                                dataX.append(a)
                                dataY.append(dataset[i + time_step, 0])
                            return np.array(dataX), np.array(dataY)

                        time_step = 15
                        X_train, y_train = create_dataset(train_data, time_step)
                        X_test, y_test = create_dataset(test_data, time_step)

                        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

                        
                        time_step = 15
                        learning_rate = 0.006120811879952696
                        num_lstm_layers = 1
                        lstm_units = 16
                        batch_size = 10
                        num_epochs = 100

                        model = Sequential()
                        model.add(LSTM(units=lstm_units, input_shape=(time_step, 1)))
                        for _ in range(num_lstm_layers - 1):
                            model.add(LSTM(units=lstm_units))
                        model.add(Dense(units=1))

                        optimizer = Adam(learning_rate=learning_rate)
                        model.compile(optimizer=optimizer, loss='mean_squared_error')

                        model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=1)

                        loss = model.evaluate(X_test, y_test)

                        train_predict = model.predict(X_train)
                        test_predict = model.predict(X_test)

                        train_predict = scaler.inverse_transform(train_predict)
                        test_predict = scaler.inverse_transform(test_predict)
                        original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
                        original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

                        look_back = time_step

            
                        trainPredictPlot = np.empty_like(argus)
                        trainPredictPlot[:, :] = np.nan
                        trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

                        testPredictPlot = np.empty_like(argus)
                        testPredictPlot[:, :] = np.nan
                        testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(argus) - 1, :] = test_predict
                        
            
                        plotdf = pd.DataFrame({
                            'Start_date': copy_price['Start_date'],
                            'original_price': copy_price['Argus_High'],
                            'train_predicted': trainPredictPlot.reshape(1, -1)[0].tolist(),
                            'test_predicted': testPredictPlot.reshape(1, -1)[0].tolist()
                        })

                        fig = px.line(plotdf, x=plotdf['Start_date'],
                                    y=[plotdf['original_price'], plotdf['train_predicted'], plotdf['test_predicted']],
                                    labels={'value': 'price', '': 'Date'})
                        fig.update_layout(title_text='',
                                        plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='')

                        names = cycle(['Harga Aktual', 'Train predicted price', 'Test predicted price'])
                        fig.for_each_trace(lambda t: t.update(name=next(names), line_width=4,))

                        fig.update_xaxes(showgrid=False)
                        fig.update_yaxes(showgrid=False)

                        st.markdown(
                            f'<style> .css-1ksj3el {{ width: 90%; }} </style>',
                            unsafe_allow_html=True
                        )
                        st.markdown("<h1 style='text-align:center'>Train Test Plot</h1>", unsafe_allow_html=True)
                        st.plotly_chart(fig, use_container_width=True)


                        st.markdown("<h1 style='text-align:center'>Model Evaluation Metrics</h1>", unsafe_allow_html=True)

                        train_rmse = math.sqrt(mean_squared_error(original_ytrain, train_predict))
                        train_mse = mean_squared_error(original_ytrain, train_predict)
                        train_mae = mean_absolute_error(original_ytrain, train_predict)
                        train_r2 = r2_score(original_ytrain, train_predict)

                        test_rmse = math.sqrt(mean_squared_error(original_ytest, test_predict))
                        test_mse = mean_squared_error(original_ytest, test_predict)
                        test_mae = mean_absolute_error(original_ytest, test_predict)
                        test_r2 = r2_score(original_ytest, test_predict)


                        db_config = {
                            'host': '217.21.73.32',
                            'user': 'u1801671_dmbsupredict',
                            'password': 'r.Sv[q!{=gl6',
                            'database': 'u1801671_dmbsupredict'
                        }

                        db_connection = mysql.connector.connect(**db_config)
                        cursor = db_connection.cursor()

                        try:
                            cursor.execute("DELETE FROM lstm_r2score_high")
                            cursor.execute("DELETE FROM lstm_maescore_high")
                            cursor.execute("DELETE FROM lstm_msescore_high")
                            cursor.execute("DELETE FROM lstm_rmsescore")

                            cursor.execute("INSERT INTO lstm_r2score_high (train, test) VALUES (%s, %s)", (train_r2, test_r2))
                            cursor.execute("INSERT INTO lstm_maescore_high (train, test) VALUES (%s, %s)", (train_mae, test_mae))
                            cursor.execute("INSERT INTO lstm_msescore_high (train, test) VALUES (%s, %s)", (train_mse, test_mse))
                            cursor.execute("INSERT INTO lstm_rmsescore_high (train, test) VALUES (%s, %s)", (train_rmse, test_rmse))

                            db_connection.commit()

                        except mysql.connector.Error as err:
                            print(f"Error: {err}")
                            db_connection.rollback()

                        finally:
                            cursor.close()
                            db_connection.close()

                        
                        coll1, coll2 = st.columns(2)
                        
                        with coll1:
                            fig = go.Figure()

                            fig.add_trace(go.Bar(
                                x=['Training R-squared', 'Testing R-squared'],
                                y=[train_r2, test_r2],
                                text=[round(train_r2, 2), round(test_r2, 2)],
                                textposition='auto',
                                marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
                            ))

                    
                            fig.update_layout(
                                title='R-squared Score',
                                xaxis_title='',
                                yaxis_title='',
                                title_x=0.5,  
                            )

                            st.plotly_chart(fig)

                        with coll2:
                            fig = go.Figure()

                            fig.add_trace(go.Bar(
                                x=['Training MAE', 'Testing MAE'],
                                y=[train_mae, test_mae],
                                text=[round(train_mae, 2), round(test_mae, 2)],
                                textposition='auto',
                                marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
                            ))

                            fig.update_layout(
                                title='MAE Score',
                                xaxis_title='',
                                yaxis_title='',
                                title_x=0.5, 
                            
                            )

                
                            st.plotly_chart(fig)

                        colll1, colll2 = st.columns(2)
                        
                        with colll1:
                            fig = go.Figure()

                            fig.add_trace(go.Bar(
                                x=['Training MSE', 'Testing MSE'],
                                y=[train_mse, test_mse],
                                text=[round(train_mse, 2), round(test_mse, 2)],
                                textposition='auto',
                                marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
                            ))

                
                            fig.update_layout(
                                title='MSE Score',
                                xaxis_title='',
                                yaxis_title='',
                                title_x=0.5, 
                            
                            )


                            st.plotly_chart(fig)

                        with colll2:
                            fig = go.Figure()

                            fig.add_trace(go.Bar(
                                x=['Training RMSE', 'Testing RMSE'],
                                y=[train_rmse, test_rmse],
                                text=[round(train_rmse, 2), round(test_rmse, 2)],
                                textposition='auto',
                                marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
                            ))

                            fig.update_layout(
                                title='RMSE Score',
                                xaxis_title='',
                                yaxis_title='',
                                title_x=0.5, 
                                
                            )

                    
                            st.plotly_chart(fig)



                        

                        
                        st.write('Col 2:')
                        st.subheader("Train Metrics:")
                        st.write(f"RMSE: {train_rmse:.2f}")
                        st.write(f"MSE: {train_mse:.2f}")
                        st.write(f"MAE: {train_mae:.2f}")
                        st.write(f"R-squared: {train_r2:.2f}")

                        st.subheader("Test Metrics:")
                        st.write(f"RMSE: {test_rmse:.2f}")
                        st.write(f"MSE: {test_mse:.2f}")
                        st.write(f"MAE: {test_mae:.2f}")
                        st.write(f"R-squared: {test_r2:.2f}")

            
                        st.subheader("Prediction Results")
                        st.write("Test Loss:", loss)



            
                        x_input = test_data[len(test_data) - time_step:].reshape(1, -1)
                        temp_input = list(x_input)
                        temp_input = temp_input[0].tolist()

                        lst_output = []
                        n_steps = time_step
                        i = 0
                        pred_week = pred_week

                        
                        while i < pred_week:
                            if len(temp_input) > time_step:
                                x_input = np.array(temp_input[1:])
                                x_input = x_input.reshape(1, -1)
                                x_input = x_input.reshape((1, n_steps, 1))

                                yhat = model.predict(x_input, verbose=0)
                                temp_input.extend(yhat[0].tolist())
                                temp_input = temp_input[1:]
                                lst_output.extend(yhat.tolist())

                                i = i + 1
                            else:
                                x_input = x_input.reshape((1, n_steps, 1))
                                yhat = model.predict(x_input, verbose=0)
                                temp_input.extend(yhat[0].tolist())
                                lst_output.extend(yhat.tolist())

                                i = i + 1

                        
                        start_date = pd.to_datetime(start_date)
                        end_date = pd.to_datetime(end_date)

                        Start_date = df['Start_date'].sort_values(ascending=False).iloc[16].strftime('%Y-%m-%d')

                        
                        last_week=np.arange(1,time_step+1)
                        day_pred=np.arange(time_step+1,time_step+pred_week+1)
                        print(last_week)
                        print(day_pred)

                        temp_mat = np.empty((len(last_week)+pred_week+1,1))
                        temp_mat[:] = np.nan
                        temp_mat = temp_mat.reshape(1,-1).tolist()[0]
                        

                        last_original_week_value = temp_mat
                        next_predicted_week_value = temp_mat

                        last_original_week_value[0:time_step+1] = scaler.inverse_transform(argus[len(argus)-time_step:]).reshape(1,-1).tolist()[0]
                        
                        next_predicted_week_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]
                        
                        conv1dgru_results = {
                            'last_original_week_value': last_original_week_value,
                            'next_predicted_week_value': next_predicted_week_value,
                        }

                        new_pred_plot = pd.DataFrame({
                            'last_original_week_value':last_original_week_value,
                            'next_predicted_week_value':next_predicted_week_value,
                            
                        })

                        names = cycle(['Last 15 week close price','Predicted next 10 week price'])
                        new_pred_plot['Timestamp'] = pd.date_range(start=Start_date, periods=len(last_week)+pred_week+1, freq='w')


            
                        st.markdown("<h1 style='text-align:center'>Plot Prediction</h1>", unsafe_allow_html=True)
                        fig = px.line(new_pred_plot, x='Timestamp', y=['last_original_week_value', 'next_predicted_week_value'],
                                    labels={'value': 'Stock price'},
                                    title='Plot Prediction')

                        fig.update_layout(plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')

                        st.markdown(
                            f'<style> .css-1ksj3el {{ width: 90%; }} </style>',
                            unsafe_allow_html=True
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        
                        timestamps = pd.date_range(start=Start_date, periods=len(last_week) + pred_week + 1, freq='w')
                        prediction_results = pd.DataFrame({
                            'Timestamp': timestamps,
                            'Predicted next 10 week price': next_predicted_week_value
                        })

                        if 'last_week' in prediction_results:
                            prediction_results.drop('last_week', axis=1, inplace=True)

                        prediction_results.dropna(subset=['Predicted next 10 week price'], inplace=True)
                        st.write(prediction_results)

                        

            
                        db_connection = mysql.connector.connect(
                            host="217.21.73.32",
                            user="u1801671_dmbsupredict",
                            password="r.Sv[q!{=gl6",
                            database="u1801671_dmbsupredict"
                        )

                
                        cursor = db_connection.cursor()
                        cursor.execute("TRUNCATE TABLE lstm_predict_high")
                
                        for index, row in prediction_results.iterrows():
                            cursor.execute("INSERT INTO lstm_predict_high (Timestamp, `Predicted_price`) VALUES (%s, %s)",
                                        (row['Timestamp'], row['Predicted next 10 week price']))

                
                        db_connection.commit()

                    
                        cursor.close()
                        db_connection.close()


                else:
                    st.error("Failed to connect to the database. Check your database connection settings.")

       
        
        



        if selected_data == 'Argus_Low':
            if st.button("Train"):
                if connection.is_connected():
                    cursor = connection.cursor(dictionary=True)
                    cursor.execute("SELECT * FROM harga_argus")
                    data = cursor.fetchall()
                    df = pd.DataFrame(data)
                    cursor.close()
                    connection.close()

                    with st.spinner("In Progress..."):
                        trans = ['Start_date', 'End_date']

                        for column in trans:
                            df[column] = df[column].astype(str)
                            df[column] = df[column].str.replace(r'\s+', '', regex=True)
                            df[column] = df[column].str[:4] + '-' + df[column].str[4:7] + df[column].str[7:]

                        df['Start_date'] = pd.to_datetime(df['Start_date'])
                        df['End_date'] = pd.to_datetime(df['End_date'])

                        argus = df[['Start_date', 'Argus_Low']]

                        copy_price = argus.copy()
                        del argus['Start_date']
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        argus = scaler.fit_transform(np.array(argus).reshape(-1, 1))

                        training_size = int(len(argus) * 0.65)
                        test_size = len(argus) - training_size
                        train_data, test_data = argus[0:training_size, :], argus[training_size:len(argus), :1]

                        def create_dataset(dataset, time_step=1):
                            dataX, dataY = [], []
                            for i in range(len(dataset) - time_step - 1):
                                a = dataset[i:(i + time_step), 0]
                                dataX.append(a)
                                dataY.append(dataset[i + time_step, 0])
                            return np.array(dataX), np.array(dataY)

                        time_step = 15
                        X_train, y_train = create_dataset(train_data, time_step)
                        X_test, y_test = create_dataset(test_data, time_step)

                        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

                        
                        time_step = 15
                        learning_rate = 0.006120811879952696
                        num_lstm_layers = 1
                        lstm_units = 16
                        batch_size = 10
                        num_epochs = 100

                        model = Sequential()
                        model.add(LSTM(units=lstm_units, input_shape=(time_step, 1)))
                        for _ in range(num_lstm_layers - 1):
                            model.add(LSTM(units=lstm_units))
                        model.add(Dense(units=1))

                        optimizer = Adam(learning_rate=learning_rate)
                        model.compile(optimizer=optimizer, loss='mean_squared_error')

                        model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=1)

                        loss = model.evaluate(X_test, y_test)

                        train_predict = model.predict(X_train)
                        test_predict = model.predict(X_test)

                        train_predict = scaler.inverse_transform(train_predict)
                        test_predict = scaler.inverse_transform(test_predict)
                        original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
                        original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

                        look_back = time_step

                        trainPredictPlot = np.empty_like(argus)
                        trainPredictPlot[:, :] = np.nan
                        trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

                        testPredictPlot = np.empty_like(argus)
                        testPredictPlot[:, :] = np.nan
                        testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(argus) - 1, :] = test_predict
                        
                        plotdf = pd.DataFrame({
                            'Start_date': copy_price['Start_date'],
                            'original_price': copy_price['Argus_Low'],
                            'train_predicted': trainPredictPlot.reshape(1, -1)[0].tolist(),
                            'test_predicted': testPredictPlot.reshape(1, -1)[0].tolist()
                        })

                        fig = px.line(plotdf, x=plotdf['Start_date'],
                                    y=[plotdf['original_price'], plotdf['train_predicted'], plotdf['test_predicted']],
                                    labels={'value': 'price', '': 'Date'})
                        fig.update_layout(title_text='',
                                        plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='')

                        names = cycle(['Harga Aktual', 'Train predicted price', 'Test predicted price'])
                        fig.for_each_trace(lambda t: t.update(name=next(names), line_width=4,))

                        fig.update_xaxes(showgrid=False)
                        fig.update_yaxes(showgrid=False)

                        st.markdown(
                            f'<style> .css-1ksj3el {{ width: 90%; }} </style>',
                            unsafe_allow_html=True
                        )
                        st.markdown("<h1 style='text-align:center'>Train Test Plot</h1>", unsafe_allow_html=True)
                        st.plotly_chart(fig, use_container_width=True)


                        st.markdown("<h1 style='text-align:center'>Model Evaluation Metrics</h1>", unsafe_allow_html=True)

                        train_rmse = math.sqrt(mean_squared_error(original_ytrain, train_predict))
                        train_mse = mean_squared_error(original_ytrain, train_predict)
                        train_mae = mean_absolute_error(original_ytrain, train_predict)
                        train_r2 = r2_score(original_ytrain, train_predict)

                        test_rmse = math.sqrt(mean_squared_error(original_ytest, test_predict))
                        test_mse = mean_squared_error(original_ytest, test_predict)
                        test_mae = mean_absolute_error(original_ytest, test_predict)
                        test_r2 = r2_score(original_ytest, test_predict)


                        db_config = {
                            'host': '217.21.73.32',
                            'user': 'u1801671_dmbsupredict',
                            'password': 'r.Sv[q!{=gl6',
                            'database': 'u1801671_dmbsupredict'
                        }

                        db_connection = mysql.connector.connect(**db_config)
                        cursor = db_connection.cursor()

                        try:
                            cursor.execute("DELETE FROM lstm_r2score_low")
                            cursor.execute("DELETE FROM lstm_maescore_low")
                            cursor.execute("DELETE FROM lstm_msescore_low")
                            cursor.execute("DELETE FROM lstm_rmsescore_low")

                            cursor.execute("INSERT INTO lstm_r2score_low (train, test) VALUES (%s, %s)", (train_r2, test_r2))
                            cursor.execute("INSERT INTO lstm_maescore_low (train, test) VALUES (%s, %s)", (train_mae, test_mae))
                            cursor.execute("INSERT INTO lstm_msescore_low (train, test) VALUES (%s, %s)", (train_mse, test_mse))
                            cursor.execute("INSERT INTO lstm_rmsescore_low (train, test) VALUES (%s, %s)", (train_rmse, test_rmse))

                            db_connection.commit()

                        except mysql.connector.Error as err:
                            print(f"Error: {err}")
                            db_connection.rollback()

                        finally:
                            cursor.close()
                            db_connection.close()

                        
                        coll1, coll2 = st.columns(2)
                        
                        with coll1:
                            fig = go.Figure()

                            fig.add_trace(go.Bar(
                                x=['Training R-squared', 'Testing R-squared'],
                                y=[train_r2, test_r2],
                                text=[round(train_r2, 2), round(test_r2, 2)],
                                textposition='auto',
                                marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
                            ))

                            fig.update_layout(
                                title='R-squared Score',
                                xaxis_title='',
                                yaxis_title='',
                                title_x=0.5,  
                            )

                        
                            st.plotly_chart(fig)

                        with coll2:
                            fig = go.Figure()

                            fig.add_trace(go.Bar(
                                x=['Training MAE', 'Testing MAE'],
                                y=[train_mae, test_mae],
                                text=[round(train_mae, 2), round(test_mae, 2)],
                                textposition='auto',
                                marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
                            ))

                            fig.update_layout(
                                title='MAE Score',
                                xaxis_title='',
                                yaxis_title='',
                                title_x=0.5, 
                            
                            )

                            st.plotly_chart(fig)

                        colll1, colll2 = st.columns(2)
                        
                        with colll1:
                            fig = go.Figure()

                            fig.add_trace(go.Bar(
                                x=['Training MSE', 'Testing MSE'],
                                y=[train_mse, test_mse],
                                text=[round(train_mse, 2), round(test_mse, 2)],
                                textposition='auto',
                                marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
                            ))

                            fig.update_layout(
                                title='MSE Score',
                                xaxis_title='',
                                yaxis_title='',
                                title_x=0.5,  
                            
                            )

                            st.plotly_chart(fig)

                        with colll2:
                            fig = go.Figure()

                            fig.add_trace(go.Bar(
                                x=['Training RMSE', 'Testing RMSE'],
                                y=[train_rmse, test_rmse],
                                text=[round(train_rmse, 2), round(test_rmse, 2)],
                                textposition='auto',
                                marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
                            ))

                        
                            fig.update_layout(
                                title='RMSE Score',
                                xaxis_title='',
                                yaxis_title='',
                                title_x=0.5, 
                                
                            )

                            st.plotly_chart(fig)



                    


                    
                        st.write('Col 2:')
                        st.subheader("Train Metrics:")
                        st.write(f"RMSE: {train_rmse:.2f}")
                        st.write(f"MSE: {train_mse:.2f}")
                        st.write(f"MAE: {train_mae:.2f}")
                        st.write(f"R-squared: {train_r2:.2f}")

                        st.subheader("Test Metrics:")
                        st.write(f"RMSE: {test_rmse:.2f}")
                        st.write(f"MSE: {test_mse:.2f}")
                        st.write(f"MAE: {test_mae:.2f}")
                        st.write(f"R-squared: {test_r2:.2f}")

                    
                        st.subheader("Prediction Results")
                        st.write("Test Loss:", loss)



                        x_input = test_data[len(test_data) - time_step:].reshape(1, -1)
                        temp_input = list(x_input)
                        temp_input = temp_input[0].tolist()

                        lst_output = []
                        n_steps = time_step
                        i = 0
                        pred_week = pred_week

                        
                        while i < pred_week:
                            if len(temp_input) > time_step:
                                x_input = np.array(temp_input[1:])
                                x_input = x_input.reshape(1, -1)
                                x_input = x_input.reshape((1, n_steps, 1))

                                yhat = model.predict(x_input, verbose=0)
                                temp_input.extend(yhat[0].tolist())
                                temp_input = temp_input[1:]
                                lst_output.extend(yhat.tolist())

                                i = i + 1
                            else:
                                x_input = x_input.reshape((1, n_steps, 1))
                                yhat = model.predict(x_input, verbose=0)
                                temp_input.extend(yhat[0].tolist())
                                lst_output.extend(yhat.tolist())

                                i = i + 1


                        start_date = pd.to_datetime(start_date)
                        end_date = pd.to_datetime(end_date)

                        Start_date = df['Start_date'].sort_values(ascending=False).iloc[16].strftime('%Y-%m-%d')

                        
                        last_week=np.arange(1,time_step+1)
                        day_pred=np.arange(time_step+1,time_step+pred_week+1)
                        print(last_week)
                        print(day_pred)

                        temp_mat = np.empty((len(last_week)+pred_week+1,1))
                        temp_mat[:] = np.nan
                        temp_mat = temp_mat.reshape(1,-1).tolist()[0]
                      

                        last_original_week_value = temp_mat
                        next_predicted_week_value = temp_mat

                        last_original_week_value[0:time_step+1] = scaler.inverse_transform(argus[len(argus)-time_step:]).reshape(1,-1).tolist()[0]
                        
                        next_predicted_week_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]
                        
                        conv1dgru_results = {
                            'last_original_week_value': last_original_week_value,
                            'next_predicted_week_value': next_predicted_week_value,
                        }

                        new_pred_plot = pd.DataFrame({
                            'last_original_week_value':last_original_week_value,
                            'next_predicted_week_value':next_predicted_week_value,
                            
                        })

                        names = cycle(['Last 15 week close price','Predicted next 10 week price'])
                        new_pred_plot['Timestamp'] = pd.date_range(start=Start_date, periods=len(last_week)+pred_week+1, freq='w')


                    
                        st.markdown("<h1 style='text-align:center'>Plot Prediction</h1>", unsafe_allow_html=True)
                        fig = px.line(new_pred_plot, x='Timestamp', y=['last_original_week_value', 'next_predicted_week_value'],
                                    labels={'value': 'Stock price'},
                                    title='Plot Prediction')

                        fig.update_layout(plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')

                        st.markdown(
                            f'<style> .css-1ksj3el {{ width: 90%; }} </style>',
                            unsafe_allow_html=True
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        
                        timestamps = pd.date_range(start=Start_date, periods=len(last_week) + pred_week + 1, freq='w')
                        prediction_results = pd.DataFrame({
                            'Timestamp': timestamps,
                            'Predicted next 10 week price': next_predicted_week_value
                        })

                        if 'last_week' in prediction_results:
                            prediction_results.drop('last_week', axis=1, inplace=True)

                        prediction_results.dropna(subset=['Predicted next 10 week price'], inplace=True)
                        st.write(prediction_results)

                        
                        db_connection = mysql.connector.connect(
                            host="217.21.73.32",
                            user="u1801671_dmbsupredict",
                            password="r.Sv[q!{=gl6",
                            database="u1801671_dmbsupredict"
                        )

                        cursor = db_connection.cursor()
                        cursor.execute("TRUNCATE TABLE lstm_predict_low")
            
                        for index, row in prediction_results.iterrows():
                            cursor.execute("INSERT INTO lstm_predict_low (Timestamp, `Predicted_price`) VALUES (%s, %s)",
                                        (row['Timestamp'], row['Predicted next 10 week price']))

            
                        db_connection.commit()

                        cursor.close()
                        db_connection.close()


                else:
                    st.error("Failed to connect to the database. Check your database connection settings.")

    
            