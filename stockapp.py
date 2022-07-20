import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np


app = dash.Dash()
server = app.server

scaler=MinMaxScaler(feature_range=(0,1))



df_googl = pd.read_csv(r"Google_data.csv")

new_dataset=pd.DataFrame(index=range(0,len(df_googl)),columns=['date','close'])
new_dataset["date"]=df_googl['date']
new_dataset["close"]=df_googl["close"]

new_dataset.index=new_dataset.date
new_dataset.drop("date",axis=1,inplace=True)

final_dataset=new_dataset.values

train_data=final_dataset[0:3605,:]
valid_data=final_dataset[3605:,:]

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(final_dataset)


x_train_data,y_train_data=[],[]

for i in range(60,len(train_data)):
    x_train_data.append(scaled_data[i-60:i,0])
    y_train_data.append(scaled_data[i,0])
    
x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data)

x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))

model=load_model(r"saved_lstm_model.h5")

inputs_data=new_dataset[len(new_dataset)-len(valid_data)-60:].values
inputs_data=inputs_data.reshape(-1,1)
inputs_data=scaler.transform(inputs_data)

X_test=[]
for i in range(60,inputs_data.shape[0]):
    X_test.append(inputs_data[i-60:i,0])
X_test=np.array(X_test)

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
closing_price=model.predict(X_test)
closing_price=scaler.inverse_transform(closing_price)

train_data=new_dataset[:3605]
valid_data=new_dataset[3605:]
valid_data['Predictions']=closing_price



df= pd.read_csv(r"stockprices.csv")

app.layout = html.Div([
   
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
   
    dcc.Tabs(id="tabs", children=[
       
        dcc.Tab(label='Google Stock Data',children=[
			html.Div([
				html.H2("Actual closing price",style={"textAlign": "center"}),
				dcc.Graph(
					id="Actual Data",
					figure={
						"data":[
							go.Scatter(
								x=valid_data.index,
								y=valid_data["close"],
								mode='lines'
							)

						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}

				),
				html.H2("LSTM Predicted closing price",style={"textAlign": "center"}),
				dcc.Graph(
					id="Predicted Data",
					figure={
						"data":[
							go.Scatter(
								x=valid_data.index,
								y=valid_data["Predictions"],
								mode='lines'
							)

						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}

				)				
			])        		


        ]),
        dcc.Tab(label='Stock Data Analysis', children=[
            html.Div([
                html.H1("Stocks Open vs Close", 
                        style={'textAlign': 'center'}),
              
                dcc.Dropdown(id='my-dropdown',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple','value': 'AAPL'}, 
                                      {'label': 'Meta', 'value': 'META'}, 
                                      {'label': 'Microsoft','value': 'MSFT'},
                                      {'label': 'Netflix','value': 'NFLX'},
                                      {'label': 'Adobe','value': 'ADBE'},
                                      {'label': 'Google','value': 'GOOGL'},
                                      {'label': 'Amazon','value': 'AMZN'},
                                      {'label': 'Goldman Sachs','value': 'GS'},
                                      {'label': 'General Electric','value': 'GE'},
                                      {'label': 'Cisco','value': 'CSCO'},
                                      {'label': 'IBM','value': 'IBM'},
                                      {'label': 'Walmart','value': 'WMT'},
                                      {'label': 'JP Morgan','value': 'JPM'},
                                      {'label': 'Airbnb','value': 'ABNB'}], 
                             multi=True,value=['META'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='highlow'),
                html.H1("Stocks Market Volume", style={'textAlign': 'center'}),
         
                dcc.Dropdown(id='my-dropdown2',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple','value': 'AAPL'}, 
                                      {'label': 'Meta', 'value': 'META'}, 
                                      {'label': 'Microsoft','value': 'MSFT'},
                                      {'label': 'Netflix','value': 'NFLX'},
                                      {'label': 'Adobe','value': 'ADBE'},
                                      {'label': 'Google','value': 'GOOGL'},
                                      {'label': 'Amazon','value': 'AMZN'},
                                      {'label': 'Goldman Sachs','value': 'GS'},
                                      {'label': 'General Electric','value': 'GE'},
                                      {'label': 'Cisco','value': 'CSCO'},
                                      {'label': 'IBM','value': 'IBM'},
                                      {'label': 'Walmart','value': 'WMT'},
                                      {'label': 'JP Morgan','value': 'JPM'},
                                      {'label': 'Airbnb','value': 'ABNB'}], 
                             multi=True,value=['META'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='volume')
            ], className="container"),
        ])


    ])
])







@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","META": "Meta","MSFT": "Microsoft",
                "NFLX": "Netflix", "ADBE": "Adobe", "GOOGL":"Google", "AMZN": "Amazon",
                "GS": "Goldman Sachs", "GE": "General Electric", "CSCO": "Cisco",
                "IBM": "IBM", "WMT": "Walmart", "JPM": "JP Morgan", "ABNB": "Airbnb"
               }
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
          go.Scatter(x=df[df["Company_ticker"] == stock]["date"],
                     y=df[df["Company_ticker"] == stock]["open"],
                     mode='lines', opacity=0.7, 
                     name=f'Open {dropdown[stock]}',textposition='bottom center'))
        trace2.append(
          go.Scatter(x=df[df["Company_ticker"] == stock]["date"],
                     y=df[df["Company_ticker"] == stock]["close"],
                     mode='lines', opacity=0.6,
                     name=f'Close {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#1c14fa", '#FF4F00', '#fa020b', 
                                            '#1a9906', '#bb00ff', '#FF0056'],
            height=600,
            title=f"Open and Close Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Price (USD)"})}
    return figure


@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","META": "Meta","MSFT": "Microsoft",
                "NFLX": "Netflix", "ADBE": "Adobe", "GOOGL":"Google", "AMZN": "Amazon",
                "GS": "Goldman Sachs", "GE": "General Electric", "CSCO": "Cisco",
                "IBM": "IBM", "WMT": "Walmart", "JPM": "JP Morgan", "ABNB": "Airbnb"
               }
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
          go.Scatter(x=df[df["Company_ticker"] == stock]["date"],
                     y=df[df["Company_ticker"] == stock]["volume"],
                     mode='lines', opacity=0.7,
                     name=f'Volume {dropdown[stock]}', textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data, 
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M',
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Transactions Volume"})}
    return figure



if __name__=='__main__':
	app.run_server(debug=True)

