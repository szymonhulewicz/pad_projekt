from turtle import position
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Wczytanie danych
df = pd.read_csv('messy_data.csv')

# Poprawa nazw kolumn (usunięcie spacji przed nazwami kolumn)
df.columns = df.columns.str.strip()

# 1a. Usunięcie duplikatów
df.drop_duplicates(inplace=True)

# 1b. Identyfikacja i obsługa wartości odstających
numeric_columns = ['carat', 'x dimension', 'y dimension', 'z dimension', 'depth', 'table', 'price']

# Konwersja kolumn numerycznych do float
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Usunięcie wierszy z brakującymi danymi w kolumnie 'price'
df_no_price = df.dropna(subset=['price'])

# Uzupełnienie brakujących danych w pozostałych kolumnach
df_no_price.loc[:, numeric_columns] = df_no_price[numeric_columns].fillna(df_no_price[numeric_columns].mean(numeric_only=True))

# Identyfikacja i obsługa wartości odstających
Q1 = df_no_price[numeric_columns].quantile(0.25)
Q3 = df_no_price[numeric_columns].quantile(0.75)
IQR = Q3 - Q1
df_clean = df_no_price[~((df[numeric_columns] < (Q1 - 1.2 * IQR)) | (df[numeric_columns] > (Q3 + 1.2 * IQR))).any(axis=1)]
selected_columns = [col for col in df.columns if col != 'price']



# 1e. Normalizacja skali wartości
# Normalizacja kolumn numerycznych
df_clean.loc[:, numeric_columns] = (
    df_clean[numeric_columns] - df_clean[numeric_columns].min()) / (
    df_clean[numeric_columns].max() - df_clean[numeric_columns].min())

# 4. Budowa modelu regresji ceny od pozostałych zmiennych
X = df_clean[selected_columns]
y = df_clean['price']

# Kodowanie kategorycznych danych jakościowych
X_encoded = pd.get_dummies(X, columns=['clarity', 'color', 'cut'], drop_first=True)

# Podział danych na zbiór treningowy, walidacyjny i testowy
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Budowa modelu regresji
model = LinearRegression()

# Eliminacja wsteczna (backward elimination)
selector = RFE(model, step=1)
selector = selector.fit(X_train, y_train)

# Wybór istotnych zmiennych
selected_columns = X_encoded.columns[selector.support_]

# Budowa modelu z istotnymi zmiennymi
model.fit(X_train[selected_columns], y_train)

# Ocena modelu na zbiorze testowym
y_pred = model.predict(X_test[selected_columns])
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

app = dash.Dash(__name__)

# Układ strony
app.layout = html.Div(children=[
    html.H1(children='Dashboard analizy danych i modelu regresji'),

    html.Div(children='Zależność ceny od wybranej zmiennej:'),
    
    dcc.Dropdown(
        id='variable-dropdown',
        options=[
            {'label': col, 'value': col} for col in selected_columns
        ],
        value=selected_columns[0]  
    ),

    dcc.Graph(
        id='scatter-plot',
        style={'width': '1200px',
               'height': '1000px',
               'margin-left': 'auto',
               'margin-right': 'auto'
               }     
    ),

    dcc.Graph(
        id='regression-visualization',
        figure={'data': [{'x': [x_val for x_val, y_val in zip(y_test, y_pred) if y_val > 0],
                          'y': [y_val for y_val in y_pred if y_val > 0],
                          'type': 'scatter', 'mode': 'markers'}],
                'layout': {'title': 'Wizualizacja modelu regresji',
                           'xaxis': {'title': 'Cena rzeczywista'},
                           'yaxis': {'title': 'Cena przewidywana'}}},
        style={'width': '1200px',
               'height': '1000px',
               'margin-left': 'auto',
               'margin-right': 'auto'
               }        
    ),

    html.Div([
        html.Hr(),  # Linia pozioma oddzielająca wykresy od wartości MSE i R^2
        html.H3('Wartości miar jakości modelu:'),
        html.P(f"Mean Squared Error (MSE): {mse}"),
        html.P(f"R-squared (R^2): {r2}")
    ])

])

@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('variable-dropdown', 'value')]
)
def update_scatter_plot(selected_variable):
    fig = px.scatter(df_clean, x=selected_variable, y='price', title=f'Zależność ceny od {selected_variable}')
    fig.update_layout(
        showlegend=True,
    )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)