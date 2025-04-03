# Importar Librerías (adaptación de la estrategia 7)
import yfinance as yf
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from itertools import product
from copy import deepcopy
import time
from tqdm import tqdm # Para mantener el control total del progreso de la optimización.
import warnings
warnings.filterwarnings("ignore")
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class Estrategia7_tp_sl:
    __version__ = 1.0
    
    def __init__(self, data: pd.DataFrame, df_lowest_timeframe: pd.DataFrame, **kwargs) -> None:
        
        # Atributos
        self.data = data
        self.df_lowest_timeframe = df_lowest_timeframe
        self.df_highest__timeframe = self.find_timestamp_extremum()
        self.BB = kwargs.get("BB", {})
        self.DMI = kwargs.get("DMI", {})
        self.FT = kwargs.get("FT", {})
        self.TP_SL= kwargs.get("TP_SL", {})
        self.estrategia_calculo = None
        self.direccion_mercado = None
        self.rendimiento_final_estrategia = 0.0
        self.calculo_ratios = None
        # Atributos Privados
        ...
        
    # __repr__
    def __repr__(self) -> str:
        return self.__class__.__name__ + ".class"
    
    def find_timestamp_extremum(self) -> None:
        """
        :params: data(highest timeframe OHLCV data), df_lowest_timeframe (lowest timeframe OHLCV data)
        :return: data with three new columns: Low_time (Date), High_time (Date), High_first (Boolean)
        """
        df = deepcopy(self.data)
        # Establecer nuevas columnas
        df["Low_time"] = np.nan
        df["High_time"] = np.nan
        df["First"] = np.nan
    
        # Realiza un bucle para descubrir cuál de los Take Profit y Stop Loss aparece primero
        for i in tqdm(range(len(df) - 1)):
            # Extraer valores del marco de datos del período de tiempo más bajo
            start = df.index[i]
            end = df.index[i + 1]
            row_lowest_timeframe = self.df_lowest_timeframe.loc[start:end].iloc[:-1]
    
            # Extraer la marca de tiempo del máximo y el mínimo durante el período (marco de tiempo más alto)
            try:
                high = row_lowest_timeframe["High"].idxmax()
                low = row_lowest_timeframe["Low"].idxmin()
    
                df.loc[start, "Low_time"] = low
                df.loc[start, "High_time"] = high
    
            except Exception as e:
                print(e)
                df.loc[start, "Low_time"] = start
                df.loc[start, "High_time"] = start
    
        # Descubre cuál aparece primero
        df.loc[df["High_time"] > df["Low_time"], "First"] = 1
        df.loc[df["High_time"] < df["Low_time"], "First"] = 2
        df.loc[df["High_time"] == df["Low_time"], "First"] = 0
    
        # Verificar el número de filas sin TP y SL al mismo tiempo
        percentage_garbage_row = len(df.loc[df["First"] == 0].dropna()) / len(df) * 100
    
        if percentage_garbage_row < 95:
            print(f"WARNINGS: Garbage row: {'%.2f' % percentage_garbage_row} %")
        
        # Transformar las columnas en columnas de fecha y hora
        df.High_time = pd.to_datetime(df.High_time)
        df.Low_time = pd.to_datetime(df.Low_time)
        
        # Eliminamos la última fila porque no encontramos el extremo
        df = df.iloc[:-1]
        
        # Específico de los datos actuales
        if "Date" in df.columns:
            del df["Date"]
            
        return df
          
    def calcular(self) -> dict:
        ######################### Calcular Bandas de Bollinger (BB) #########################
        data= deepcopy(self.data)
        rolling= data[self.BB.get("columna", "Close")].rolling(window= self.BB.get("longitud", 20), min_periods= self.BB.get("longitud", 20))
        data["MA"]= rolling.mean()
        calc_intermedio= self.BB.get("std_desv ", 2.0) * rolling.std(ddof= self.BB.get("ddfo", 0))
        data["BB_Up"]= data["MA"] + calc_intermedio
        data["BB_Down"]= data["MA"] - calc_intermedio
        
        BB = data[["MA", "BB_Up", "BB_Down"]]
        
        ######################### Calcular Indice de Movimiento Direccional (DMI) #########################
        # Calcular el Rango Verdadero
        High, Low = self.data["High"], self.data["Low"]
        H_minus_L = High - Low
        prev_clo = self.data["Close"].shift(periods=1)
        H_minus_PC = abs(High - prev_clo)
        L_minus_PC = abs(prev_clo - Low)
        TR = pd.Series(np.max([H_minus_L, H_minus_PC, L_minus_PC], axis=0), index=self.data.index, name="TR")
        
        # Calcular los Movimientos Direccionales (+DM y -DM)
        pre_PDM = self.data["High"].diff().dropna()
        pre_MDM = self.data["Low"].diff(periods=-1).dropna()
        plus_DM = pre_PDM.where((pre_PDM > pre_MDM.values) & (pre_PDM > 0), 0)
        minus_DM = pre_MDM.where((pre_MDM > pre_PDM.values) & (pre_MDM > 0), 0)
        
        # Calcular los valores iniciales para las sumas suavizadas de TR, +DM, -DM
        TRL = [np.nansum(TR[:self.DMI.get("suavizado_ADX", 14) + 1])]
        PDML = [plus_DM[:self.DMI.get("suavizado_ADX", 14)].sum()]
        MDML = [minus_DM[:self.DMI.get("suavizado_ADX", 14)].sum()]
        factor = 1 - 1 / self.DMI.get("suavizado_ADX", 14)

        # Calcular las sumas suavizadas de TR, +DM y -DM utilizando el método Wilder
        for i in range(0, int(self.data.shape[0] - self.DMI.get("suavizado_ADX", 14) - 1)):
            TRL.append(TRL[i] * factor + TR[self.DMI.get("suavizado_ADX", 14) + i + 1])
            PDML.append(PDML[i] * factor + plus_DM[self.DMI.get("suavizado_ADX", 14) + i])
            MDML.append(MDML[i] * factor + minus_DM[self.DMI.get("suavizado_ADX", 14) + i])
            
        # Calcular los Indicadores Direccionales (+DI y -DI)
        PDI = np.array(PDML) / np.array(TRL) * 100
        MDI = np.array(MDML) / np.array(TRL) * 100
        # Calcular el Indice Direccional (DX)
        DX = np.abs(PDI - MDI) / (PDI + MDI) * 100
        ADX = [DX[:self.DMI.get("suavizado_ADX", 14)].mean()]
        
        # Calcular el Índice Direccional Promedio (ADX) utilizando la longitud_DI
        _ = [ADX.append((ADX[i] * (self.DMI.get("longitud_DI", 14) - 1) + DX[self.DMI.get("longitud_DI", 14) + i])/self.DMI.get("longitud_DI", 14)) for i in range(int(len(DX) - self.DMI.get("longitud_DI", 14)))]
        ADXI = pd.DataFrame(PDI, columns=["+DI"], index=self.data.index[-len(PDI):])
        ADXI["-DI"] = MDI
        ADX = pd.DataFrame(ADX, columns=["ADX"], index=self.data.index[-len(ADX):])
        
        DMI = ADX.merge(ADXI, how="outer", left_index=True, right_index=True)
    
        ######################### Calcular Indicador de Fisher Transform (FT) #########################
        # Calcular el máximo y mínimo en la ventana 'periodo'
        min_val = self.data["Close"].rolling(window=self.FT.get("longitud", 9)).min()
        max_val = self.data["Close"].rolling(window=self.FT.get("longitud", 9)).max()

        # Normalizar en el rango [-1, 1]
        rango = (self.data["Close"] - min_val) / (max_val - min_val)
        rango = rango * 2 - 1

        # Aplicar el límite para evitar valores extremos
        rango = np.clip(rango, -0.999, 0.999)

        # Aplicar la transformación Fisher
        fisher = 0.5 * np.log((1 + rango) / (1 - rango))
        fisher = pd.DataFrame(fisher)
        fisher.columns = ["FT"]
        # Suavizar la serie
        fisher = fisher.ewm(span=1, adjust=False).mean()

        FT = fisher
        
        #################################################################################
        # Guardar los cálculos
        self.estrategia_calculo = {"BB": BB, "DMI": DMI, "FT": FT}
        
        # Generar Señales
        buy_signals = ((DMI["ADX"] > 25) & (self.data["Close"] < BB["MA"]) & (FT["FT"] > 0)).astype(int)
        sell_signals = ((DMI["ADX"] < 25) & (self.data["Close"] > BB["MA"]) & (FT["FT"] < 0)).astype(int) * -1
        
        # Almacenar la dirección del mercado
        direccion = buy_signals + sell_signals
        self.direccion_mercado = direccion
        
        # Detectar Tendencia Actual
        if direccion.iloc[-1] == 1:
            tendencia_actual = {"tendencia_actual": "alcista"}
        elif direccion.iloc[-1] == -1:
            tendencia_actual = {"tendencia_actual": "bajista"}
        else:
            tendencia_actual = False
        
        return tendencia_actual
    
    def run_tp_sl(self):
        
        direccion_mercado = self.direccion_mercado
        direccion_mercado = pd.DataFrame(direccion_mercado, columns=["Señal"])
        
        
        self.df_highest__timeframe["Señal"] = direccion_mercado
        
        # Inicializar la columna 'duration' si no existe
        if 'duration' not in self.df_highest__timeframe.columns:
            self.df_highest__timeframe['duration'] = pd.Timedelta(0)  # Establecer valor predeterminado en 0 (Timedelta)

        buy = False
        sell = False
        self.df_highest__timeframe["returns"] = np.nan  # Inicializamos las ganancias (si no existen)
        
        for i in range(len(self.df_highest__timeframe)):

            # Extraer datos de la fila actual
            row = self.df_highest__timeframe.iloc[i]

            ######## OPEN BUY ########
            if buy == False and row["Señal"] == 1:
                buy = True
                open_buy_price = row["Open"]
                open_buy_date = row["Low_time"]  # Fecha de apertura (Low_time)

            # VERIFICAR
            if buy:
                
                var_buy_high = (row["High"] - open_buy_price) / open_buy_price
                var_buy_low = (row["Low"] - open_buy_price) / open_buy_price

                # Verificar si TP y SL se alcanzan en la misma vela
                if (var_buy_high > self.TP_SL.get("tp", 0.01)) and (var_buy_low < self.TP_SL.get("sl", -0.01)):

                    if row["Low_time"] == row["High_time"]:
                        pass
                    elif row["First"] == 2:
                        self.df_highest__timeframe.loc[row.name, "returns"] = (self.TP_SL.get("tp", 0.01) - self.TP_SL.get("cost", 0.001)) * self.TP_SL.get("leverage", 1)
                        # Calcular la duración como la diferencia entre las fechas
                        if self.df_highest__timeframe.loc[row.name, "duration"] == pd.Timedelta(0):
                            self.df_highest__timeframe.loc[row.name, "duration"] = (row["Low_time"] - open_buy_date)

                    elif row["First"] == 1:
                        self.df_highest__timeframe.loc[row.name, "returns"] = (self.TP_SL.get("sl", -0.01) - self.TP_SL.get("cost", 0.001)) * self.TP_SL.get("leverage", 1)
                        # Calcular la duración como la diferencia entre las fechas
                        if self.df_highest__timeframe.loc[row.name, "duration"] == pd.Timedelta(0):
                            self.df_highest__timeframe.loc[row.name, "duration"] = (row["Low_time"] - open_buy_date)

                    buy = False
                    open_buy_price = None
                    var_buy_high = 0
                    var_buy_low = 0
                    open_buy_date = None

                elif var_buy_high > self.TP_SL.get("tp", 0.01):
                    self.df_highest__timeframe.loc[row.name, "returns"] = (self.TP_SL.get("tp", 0.01) - self.TP_SL.get("cost", 0.001)) * self.TP_SL.get("leverage", 1)
                    # Calcular la duración como la diferencia entre las fechas
                    if self.df_highest__timeframe.loc[row.name, "duration"] == pd.Timedelta(0):
                        self.df_highest__timeframe.loc[row.name, "duration"] = (row["Low_time"] - open_buy_date)
                    buy = False
                    open_buy_price = None
                    var_buy_high = 0
                    var_buy_low = 0
                    open_buy_date = None
                    
                elif var_buy_low < self.TP_SL.get("sl", -0.01):
                    self.df_highest__timeframe.loc[row.name, "returns"] = (self.TP_SL.get("sl", -0.01) - self.TP_SL.get("cost", 0.001)) * self.TP_SL.get("leverage", 1)
                    # Calcular la duración como la diferencia entre las fechas
                    if self.df_highest__timeframe.loc[row.name, "duration"] == pd.Timedelta(0):
                        self.df_highest__timeframe.loc[row.name, "duration"] = (row["Low_time"] - open_buy_date)
                    buy = False
                    open_buy_price = None
                    var_buy_high = 0
                    var_buy_low = 0
                    open_buy_date = None

            ######## OPEN SELL ########
            if sell == False and row["Señal"] == -1:
                sell = True
                open_sell_price = row["Open"]
                open_sell_date = row["Low_time"]  # Fecha de apertura (Low_time)

            # VERIFICAR
            if sell:

                var_sell_high = -(row["High"] - open_sell_price) / open_sell_price
                var_sell_low = -(row["Low"] - open_sell_price) / open_sell_price

                if (var_sell_low > self.TP_SL.get("tp", 0.01)) and (var_sell_high < self.TP_SL.get("sl", -0.01)):

                    if row["Low_time"] == row["High_time"]:
                        pass
                    elif row["First"] == 1:
                        self.df_highest__timeframe.loc[row.name, "returns"] = (self.TP_SL.get("tp", 0.01) - self.TP_SL.get("cost", 0.001)) * self.TP_SL.get("leverage", 1)
                        # Calcular la duración como la diferencia entre las fechas
                        if self.df_highest__timeframe.loc[row.name, "duration"] == pd.Timedelta(0):
                            self.df_highest__timeframe.loc[row.name, "duration"] = (row["Low_time"] - open_sell_date)

                    elif row["First"] == 2:
                        self.df_highest__timeframe.loc[row.name, "returns"] = (self.TP_SL.get("sl", -0.01) - self.TP_SL.get("cost", 0.001)) * self.TP_SL.get("leverage", 1)
                        # Calcular la duración como la diferencia entre las fechas
                        if self.df_highest__timeframe.loc[row.name, "duration"] == pd.Timedelta(0):
                            self.df_highest__timeframe.loc[row.name, "duration"] = (row["Low_time"] - open_sell_date)

                    sell = False
                    open_sell_price = None
                    var_sell_high = 0
                    var_sell_low = 0
                    open_sell_date = None

                elif var_sell_low > self.TP_SL.get("tp", 0.01):
                    self.df_highest__timeframe.loc[row.name, "returns"] = (self.TP_SL.get("tp", 0.01) - self.TP_SL.get("cost", 0.001)) * self.TP_SL.get("leverage", 1)
                    # Calcular la duración como la diferencia entre las fechas
                    if self.df_highest__timeframe.loc[row.name, "duration"] == pd.Timedelta(0):
                        self.df_highest__timeframe.loc[row.name, "duration"] = (row["Low_time"] - open_sell_date)
                    sell = False
                    open_sell_price = None
                    var_sell_high = 0
                    var_sell_low = 0
                    open_sell_date = None

                elif var_sell_high < self.TP_SL.get("sl", -0.01):
                    self.df_highest__timeframe.loc[row.name, "returns"] = (self.TP_SL.get("sl", -0.01) - self.TP_SL.get("cost", 0.001)) * self.TP_SL.get("leverage", 1)
                    # Calcular la duración como la diferencia entre las fechas
                    if self.df_highest__timeframe.loc[row.name, "duration"] == pd.Timedelta(0):
                        self.df_highest__timeframe.loc[row.name, "duration"] = (row["Low_time"] - open_sell_date)
                    sell = False
                    open_sell_price = None
                    var_sell_high = 0
                    var_sell_low = 0
                    open_sell_date = None

        # Rellenar con 0 las filas sin retorno
        self.df_highest__timeframe["returns"] = self.df_highest__timeframe["returns"].fillna(value=0)
        
        return self.df_highest__timeframe
    
    def backtest_tp_sl(self):
        if 'returns' not in self.df_highest__timeframe.columns:
            raise RuntimeError("Ejecutar el método de .run_tp_sl() antes de correr el backtest")
            
        # Calcular los Retornos Diarios
        portfolio = self.df_highest__timeframe["returns"]
        portfolio = portfolio.reset_index(drop=False)  # Reseteamos para agrupar correctamente
        portfolio["Time"] = portfolio["Date"].dt.date  # Creamos una nueva columna de fecha
        portfolio = portfolio.groupby("Time")["returns"].sum()  # Agrupamos por fecha
        
        # Calcular los Retornos Acumulados
        rendimiento_acumulado = (portfolio + 1).cumprod() 
        
        self.rendimiento_final_estrategia = rendimiento_acumulado
        
        return rendimiento_acumulado
    
    def optimizar(self, bb_range: list, dmi_range: list, ft_range: list, max_iter: int = 10_000) -> pd.DataFrame:
        
        # Optimizar
        params_originales = [self.BB, self.DMI, self.FT]
        resultados = []
        combinaciones = np.array(list(product(*bb_range, *dmi_range, *ft_range)))
        
        # Seleccionar las combinaciones
        if len(combinaciones) > max_iter:
            # Elegir índices
            indices_seleccionados = np.random.choice(np.arange(0, len(combinaciones)), size=max_iter, replace=False)
            combinaciones = combinaciones[indices_seleccionados]
        for long_bb, suav_adx, long_dmi, long_ft in tqdm(combinaciones, desc="Procesando combinaciones"):
            try:
                # Reasignar Parámetros
                self.BB = {"longitud": long_bb}
                self.DMI = {"suavizado_ADX": suav_adx, "longitud_DMI": long_dmi}
                self.FT = {"longitud": long_ft}
                
                # Calcular y realizar backtest
                self.calcular()
                self.run_tp_sl()
                retorno_final = self.backtest_tp_sl()
                
                # Almacenar resultados
                resultados.append([long_bb, suav_adx, long_dmi, long_ft, retorno_final.iloc[-1]])
            except Exception as e:
                print(f"Error con combinación {long_bb, suav_adx, long_dmi, long_ft}: {e}")
        # Almacenar en un DataFrame
        resultados = pd.DataFrame(data=resultados, columns=["longitud_BB", "suavizado_ADX", "longitud_DMI", "longitud_FT", "Rendimiento"])
        resultados = resultados.sort_values(by="Rendimiento", ascending=False)
        
        # Regresar todo al estado original
        self.BB = params_originales[0]
        self.DMI = params_originales[1]
        self.FT = params_originales[2]
        self.calcular()
        self.run_tp_sl()
        self.backtest_tp_sl()
        
        return resultados
    
    def heatmap(self):
        """
        Calcula y grafica un heatmap de los retornos mensuales de la estrategia
        basándose en el rendimiento acumulado.
        """
        # Verificar que el rendimiento acumulado existe
        if self.rendimiento_final_estrategia is None or self.rendimiento_final_estrategia.empty:
            raise RuntimeError("Rendimiento final no calculado. Ejecute backtest() primero.")

        # Copiar la serie de rendimiento acumulado
        s = self.rendimiento_final_estrategia.copy()

        # Asegurar que el índice es de tipo `DatetimeIndex`
        s.index = pd.to_datetime(s.index)

        # Resample a frecuencia mensual: calcula el retorno del mes como (último/primero - 1)
        monthly_returns = s.resample('M').apply(lambda x: (x.iloc[-1] / x.iloc[0]) - 1)

        # Convertir a DataFrame y extraer año y mes
        monthly_returns_df = monthly_returns.to_frame(name="ret")
        monthly_returns_df["year"] = monthly_returns_df.index.year
        monthly_returns_df["month"] = monthly_returns_df.index.month

        # Pivotear para obtener años en filas y meses en columnas
        pivot = monthly_returns_df.pivot(index="year", columns="month", values="ret")

        # Asegurar que aparecen todas las columnas de mes (1 a 12)
        pivot = pivot.reindex(columns=np.arange(1, 13))

        # Renombrar columnas con nombres de meses
        month_names = {
            1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
            7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
        }
        pivot.rename(columns=month_names, inplace=True)

        # Convertir a porcentaje
        pivot = pivot * 100

        # Forzar el índice a enteros y ordenar
        pivot.index = pivot.index.astype(int)
        pivot = pivot.sort_index()

        # Crear el heatmap con Plotly Express
        fig = px.imshow(
            pivot,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdYlGn",
            origin="upper",
            labels={"x": "Month", "y": "Year", "color": "Return (%)"},
            title="Retornos Mensuales de la Estrategia (%)"
        )

        # Ubicar el eje x en la parte superior
        fig.update_xaxes(side="top")

        # Forzar el eje y a mostrar los años como enteros
        years = list(pivot.index)
        fig.update_yaxes(tickmode='array', tickvals=years, ticktext=[str(year) for year in years])

        fig.show()
        
    def monte_carlo(self, method="compound", n_sim=1_000):
        """
        Método para visualizar interactivamente la evolución del Monte Carlo
        y así evaluar el riesgo potencial asociado a la estrategia.
        Percentiles usados del 90% (ganancias en el mejor de los casos), y del
        10% (potenciales pérdidas máximas).

        Parámetros:
            - method : La forma de cálculo. Por defecto, "compound" (ideal por la
              naturaleza compuesta de las ganancias).
            - n_sim : Número de simulaciones de Monte Carlo. Por ejemplo, si 
              es 100, para el percentil del 1%, 1 de cada 100 realidades alternas,
              te encontrarías en el peor de los escenarios.

        """
        # Verificar que se haya calculado el rendimiento acumulado de la estrategia
        if self.rendimiento_final_estrategia is None or self.rendimiento_final_estrategia.empty:
            raise RuntimeError("Rendimiento final de la estrategia no calculado. Ejecute backtest() primero.")
        
        # Calcular los retornos diarios a partir del rendimiento acumulado de la estrategia
        returns = self.rendimiento_final_estrategia.pct_change().fillna(0).values - 1e-100

        # Generar n_sim permutaciones de los retornos diarios para las simulaciones
        random_returns = np.array([np.random.permutation(returns) for _ in tqdm(range(n_sim), desc="Simulando Monte Carlo...")])
        
        # Calcular rendimientos acumulados según el método
        if method == "simple":
            df_ret = np.cumsum(random_returns.T, axis=0) * 100
            cur_ret = np.cumsum(returns) * 100
        else:
            df_ret = (np.cumprod(1 + random_returns.T, axis=0) - 1) * 100
            cur_ret = (np.cumprod(1 + returns) - 1) * 100

        # Calcular percentiles diarios a partir de las simulaciones
        p_99, p_50, p_1 = np.percentile(df_ret, [99, 50, 1], axis=1)
        
        # Usar el índice de rendimiento acumulado para el eje x
        dates = self.rendimiento_final_estrategia.index

        # Crear la figura de Plotly
        fig = go.Figure()

        # Curva Percentile 90
        fig.add_trace(go.Scatter(
            x=dates, y=p_99,
            mode="lines",
            line=dict(color="#FF5733", dash="dash", width=2.5),
            name="Percentile 90"
        ))
        # Curva Percentile 50
        fig.add_trace(go.Scatter(
            x=dates, y=p_50,
            mode="lines",
            line=dict(color="#2ECC71", dash="dashdot", width=2.5),
            name="Percentile 50"
        ))
        # Curva Percentile 10
        fig.add_trace(go.Scatter(
            x=dates, y=p_1,
            mode="lines",
            line=dict(color="#9B59B6", dash="dot", width=2.5),
            name="Percentile 10"
        ))
        # Curva de Rendimiento Actual
        fig.add_trace(go.Scatter(
            x=dates, y=cur_ret,
            mode="lines",
            line=dict(color="dodgerblue", width=3.5),
            name="Current Returns"
        ))
        # Rellenar el área entre Percentile 90 y Percentile 10
        fig.add_trace(go.Scatter(
            x=np.concatenate([dates, dates[::-1]]),
            y=np.concatenate([p_99, p_1[::-1]]),
            fill="toself",
            fillcolor="rgba(102,159,238,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="Monte Carlo Area"
        ))
        
        fig.update_layout(
            title="Monte Carlo Simulation",
            xaxis_title="Date",
            yaxis_title="Cumulative Returns (%)",
            template="plotly_dark",
            legend=dict(orientation="h", x=0.01, y=1.15),
            width=1200,
            height=800
        )
        
        fig.show()
    
    def sharpe_rolling(self, window: int = 30) -> None:
        """
        Grafica el Sharpe Ratio rolling del portafolio/estrategia.
        Se calcula a partir de los retornos diarios derivados de self.rendimiento_final_estrategia.
        
        Parámetros:
            window: Número de días de la ventana para el cálculo rolling.
        """

        # Verificar que se haya calculado el rendimiento final
        if self.rendimiento_final_estrategia is None or self.rendimiento_final_estrategia.empty:
            raise RuntimeError("Rendimiento final no calculado. Ejecute backtest() primero.")

        # Calcular los retornos diarios a partir de la serie de rendimiento acumulado
        daily_returns = self.rendimiento_final_estrategia.pct_change().dropna()

        # Calcular el Sharpe rolling: (media rolling / std rolling) * sqrt(window)
        rolling_mean = daily_returns.rolling(window=window).mean()
        rolling_std = daily_returns.rolling(window=window).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(window)

        # Crear gráfico con Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe,
            mode='lines',
            name=f'Sharpe Ratio Rolling ({window} días)',
            line=dict(color='royalblue', width=2)
        ))
        fig.update_layout(
            title=f'Sharpe Ratio Rolling ({window} días)',
            xaxis_title='Fecha',
            yaxis_title='Sharpe Ratio',
            template='plotly_dark',
            width=1200,
            height=600
        )
        fig.show()

    def beta_rolling_tp_sl(self, window: int = 30) -> None:
        """
        Grafica la evolución del Beta Rolling de la estrategia comparado con el benchmark ^GSPC.
        """
        # Verificar que se haya calculado el rendimiento final
        if self.rendimiento_final_estrategia is None or self.rendimiento_final_estrategia.empty:
            raise RuntimeError("Rendimiento final no calculado. Ejecute backtest() primero.")

        # Calcular los retornos diarios del portafolio y normalizar el índice a fecha
        portfolio_returns = self.rendimiento_final_estrategia.pct_change().dropna()

        # Convertir el índice directamente a `DatetimeIndex`
        portfolio_returns.index = pd.to_datetime(portfolio_returns.index)

        # Asegurarse de que el índice es único
        portfolio_returns = portfolio_returns[~portfolio_returns.index.duplicated(keep='first')]

        # Definir el rango de fechas basado en el portafolio
        start = portfolio_returns.index.min().strftime("%Y-%m-%d")
        end = portfolio_returns.index.max().strftime("%Y-%m-%d")

        # Descargar datos del benchmark (^GSPC) y normalizar su índice a fecha
        benchmark_data = yf.download("^GSPC", start=start, end=end)["Close"]
        benchmark_returns = benchmark_data.pct_change().dropna()

        # Convertir el índice de `benchmark_returns` a `DatetimeIndex`
        benchmark_returns.index = pd.to_datetime(benchmark_returns.index)

        # Asegurar que el índice es único
        benchmark_returns = benchmark_returns[~benchmark_returns.index.duplicated(keep='first')]

        # Realizar un inner join para obtener únicamente las fechas comunes
        df_merged = pd.concat([portfolio_returns, benchmark_returns], axis=1, join="inner")
        df_merged.columns = ["Portfolio", "Benchmark"]
        df_merged.dropna(inplace=True)

        if df_merged.empty:
            raise RuntimeError("No se encontraron datos comunes entre el portafolio y el benchmark.")

        # Calcular el Beta Rolling:
        rolling_cov = df_merged["Portfolio"].rolling(window=window).cov(df_merged["Benchmark"])
        rolling_var = df_merged["Benchmark"].rolling(window=window).var()
        rolling_beta = rolling_cov / rolling_var

        # Crear gráfico de la evolución del Beta Rolling con Plotly (estilo QuantConnect)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rolling_beta.index,
            y=rolling_beta,
            mode="lines",
            name=f'Beta Rolling ({window} días)',
            line=dict(color="firebrick", width=2)
        ))
        # Línea horizontal de referencia en beta = 1
        fig.add_shape(
            type="line",
            x0=rolling_beta.index.min(),
            y0=1,
            x1=rolling_beta.index.max(),
            y1=1,
            line=dict(color="white", dash="dash"),
        )
        fig.update_layout(
            title=f"Evolución del Beta Rolling ({window} días) vs ^GSPC",
            xaxis_title="Fecha",
            yaxis_title="Beta",
            template="plotly_dark",
            width=1200,
            height=600
        )
        fig.show()

        def plot_drawdown(self) -> None:
            """
            Grafica el Drawdown de la estrategia.
            Se utiliza la serie de rendimiento acumulado (self.rendimiento_final_estrategia) para calcular el drawdown.
            """

            # Verificar que se haya calculado el rendimiento final
            if self.rendimiento_final_estrategia is None or self.rendimiento_final_estrategia.empty:
                raise RuntimeError("Rendimiento final no calculado. Ejecute backtest() primero.")

            # Calcular la serie de retornos acumulados y el drawdown
            cum_rets = self.rendimiento_final_estrategia.copy()
            running_max = cum_rets.cummax()
            drawdown = (cum_rets / running_max - 1)

            # Crear el gráfico con Plotly replicando el estilo solicitado
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown * 100,
                mode="lines",
                line=dict(color="#930303", width=1.5),
                fill='tozeroy',
                fillcolor="#CE5151",
                name="Drawdown %"
            ))

            fig.update_layout(
                title={"text": "DRAWDOWN", "font": {"size": 15}},
                xaxis=dict(
                    title="Fecha",
                    tickfont=dict(size=15, family="Arial", color="white"),
                ),
                yaxis=dict(
                    title="Drawdown %",
                    tickfont=dict(size=15, family="Arial", color="white"),
                ),
                template="plotly_dark",
                width=1500,
                height=800,
                showlegend=False
            )
            fig.show()

    def plot_drawdown(self) -> None:
        """
        Grafica el Drawdown de la estrategia.
        Se utiliza la serie de rendimiento acumulado (self.rendimiento_final_estrategia) para calcular el drawdown.
        """

        # Verificar que se haya calculado el rendimiento final
        if self.rendimiento_final_estrategia is None or self.rendimiento_final_estrategia.empty:
            raise RuntimeError("Rendimiento final no calculado. Ejecute backtest() primero.")

        # Calcular la serie de retornos acumulados y el drawdown
        cum_rets = self.rendimiento_final_estrategia.copy()
        running_max = cum_rets.cummax()
        drawdown = (cum_rets / running_max - 1)

        # Crear el gráfico con Plotly replicando el estilo solicitado
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown * 100,
            mode="lines",
            line=dict(color="#930303", width=1.5),
            fill='tozeroy',
            fillcolor="#CE5151",
            name="Drawdown %"
        ))

        fig.update_layout(
            title={"text": "DRAWDOWN", "font": {"size": 15}},
            xaxis=dict(
                title="Fecha",
                tickfont=dict(size=15, family="Arial", color="white"),
            ),
            yaxis=dict(
                title="Drawdown %",
                tickfont=dict(size=15, family="Arial", color="white"),
            ),
            template="plotly_dark",
            width=1500,
            height=800,
            showlegend=False
        )
        fig.show()

    def ratios_tp_sl(self, ben: str="^GSPC", timeframe: int=252):
        
        # Cálculo del TRADE LIFETIME (tiempo de vida)
        sum_dates = self.df_highest__timeframe["duration"]
        seconds = np.round(np.mean(list(sum_dates.loc[sum_dates != 0])).total_seconds())
        minutes = seconds // 60
        minutes_left = int(minutes % 60)
        hours = int(minutes // 60)
        
        # Cálculo de los Retornos Diarios
        portfolio = self.df_highest__timeframe["returns"]
        portfolio = portfolio.reset_index(drop=False)  # Reseteamos para agrupar correctamente
        portfolio["Time"] = portfolio["Date"].dt.date  # Creamos una nueva columna de fecha
        portfolio = portfolio.groupby("Time")["returns"].sum()  # Agrupamos por fecha

        ######################### Cálculo del Beta ##################################
        # Importation of benchmark
        benchmark = yf.download(ben)["Close"].pct_change(1).dropna()

        # Concat the asset and the benchmark
        join = pd.concat((portfolio, benchmark), axis=1).dropna()
        join.columns = ["returns", "benchmark"]

        # Covariance between the asset and the benchmark
        cov = np.cov(join["returns"], join["benchmark"])[0][1]

        # Compute the variance of the benchmark
        var = np.cov(join["returns"], join["benchmark"])[1][1]

        beta = cov / var

        ######################### Cálculo del Alpha #################################
        # Mean of returns for the asset
        mean_stock_return = join["returns"].mean() * timeframe

        # Mean of returns for the market
        mean_market_return = join["benchmark"].mean() * timeframe

        # Alpha
        alpha = mean_stock_return - beta * mean_market_return

        ######################### Cálculo del Ratio de Sharpe ########################
        mean = portfolio.mean() * timeframe
        std = portfolio.std() * np.sqrt(timeframe)
        sharpe = mean / std

        ######################### Cálculo del Ratio de Sortino #######################
        downside_std = portfolio[portfolio < 0].std() * np.sqrt(timeframe)
        sortino = mean / downside_std
        
        ######################### Cálculo del Drawdown ###############################
        # Compute the cumulative product returns
        coef_rets = (portfolio + 1).cumprod()
        cum_rets = (coef_rets - 1)[-1] * 100

        # Compute the running max
        running_max = np.maximum.accumulate(coef_rets)

        # Compute the drawdown
        drawdown = (coef_rets / running_max) - 1
        min_drawdon = (-drawdown.min())

        ######################### Cálculo del VaR (MC) ###############################
        theta = 0.01
        # Number of simulations
        n = 100000

        # Find the values for theta% error threshold
        t = int(n * theta)

        # Create a vector with n simulations of the normal law
        vec = pd.DataFrame(np.random.normal(mean, std, size=(n,)), columns=["Simulations"])

        # Order the values and find the theta% value
        VaR = -vec.sort_values(by="Simulations").iloc[t].values[0]

        ######################### Cálculo del cVaR (MC) ##############################
        cVaR = -vec.sort_values(by="Simulations").iloc[0:t, :].mean().values[0]

        ######################### TIME UNDERWATER ####################################
        tuw = len(drawdown[drawdown < 0]) / len(drawdown)
        
        ######################### Métricas Adicionales ###############################
        # Número de Trades
        num_trades = len(self.df_highest__timeframe[self.df_highest__timeframe["Señal"] == 1]) + len(self.df_highest__timeframe[self.df_highest__timeframe["Señal"] == -1])
        # Ratio Long/ Short
        long_trades = len(self.df_highest__timeframe[self.df_highest__timeframe["Señal"] == 1])
        short_trades = len(self.df_highest__timeframe[self.df_highest__timeframe["Señal"] == -1])
        long_short_ratio = long_trades / short_trades if short_trades > 0 else np.nan
        # Hit Ratio
        winning_trades = len(self.df_highest__timeframe[self.df_highest__timeframe["returns"] > 0])
        hit_ratio = winning_trades / num_trades if num_trades > 0 else 0
        # Profit Factor
        gross_profit = self.df_highest__timeframe[self.df_highest__timeframe["returns"] > 0]["returns"].sum()
        gross_loss = abs(self.df_highest__timeframe[self.df_highest__timeframe["returns"] < 0]["returns"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan
        # Riesgo de Ruina
        risk_of_ruin = np.round((1 - hit_ratio) ** num_trades, 2) if hit_ratio > 0 else 0
        ######################### APENDIZAR RATIOS #################################
        data = {
            "Cumulative Returns (%)": np.round(cum_rets, 2),
            "Sharpe Ratio": np.round(sharpe, 3),
            "Sortino Ratio": np.round(sortino, 3),
            "Beta": np.round(beta, 3),
            "Alpha (%)": np.round(alpha * 100, 2),
            "Average Trade Lifetime": f"{hours}H {minutes_left}min",
            "VaR (%)": np.round(VaR * 100, 2),
            "cVaR (%)": np.round(cVaR * 100, 2),
            "TUW (%)": np.round(tuw * 100, 2),
            "Drawdown (%)": np.round(min_drawdon * 100, 2),
            "Num Trades": num_trades,
            "Hit Ratio (%)": np.round(hit_ratio * 100, 2),
            "Profit Factor": np.round(profit_factor, 2),
            "Long/Short Ratio": np.round(long_short_ratio, 2),
            "Risk of Ruin (%)": np.round(risk_of_ruin * 100, 2)
        }
        # Convertir el diccionario en un DataFrame
        calculo_ratios = pd.DataFrame(list(data.items()), columns=["Métrica", "Valor"])
        
        self.calculo_ratios = calculo_ratios
        
        return self.calculo_ratios
    
    def plot(self):
        # Asegurar que los índices sean DatetimeIndex y estén ordenados
        df = self.data.copy()  # Copia para evitar modificar el original
        df = df.sort_index()
        df.index = pd.to_datetime(df.index)  # ✅ Corregido

        rendimiento = self.rendimiento_final_estrategia.fillna(1).sort_index()
        rendimiento.index = pd.to_datetime(rendimiento.index)

        # Crear subplots: Rendimiento arriba y velas abajo
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,  # Espacio entre subplots
            row_heights=[0.3, 0.7],
            subplot_titles=(
                f"Rendimiento Acumulado ({df.index[0].strftime('%Y/%m/%d')} - {df.index[-1].strftime('%Y/%m/%d')})",
                "Gráfico de Velas"
            )
        )

        # --- Subplot 1: Rendimiento Acumulado ---
        fig.add_trace(
            go.Scatter(
                x=rendimiento.index,
                y=rendimiento,
                mode="lines",
                name="Rendimiento Acumulado",
                line=dict(color="cyan", width=2)
            ),
            row=1, col=1
        )

        # --- Subplot 2: Gráfico de Velas ---
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Velas Japonesas",
            increasing_line_color="green",
            decreasing_line_color="red"
        ), row=2, col=1)

        # --- Configuración de estilo ---
        fig.update_xaxes(showline=True, linewidth=2, linecolor="white", row=2, col=1)
        fig.update_yaxes(showline=True, linewidth=2, linecolor="white", row=1, col=1)
        fig.update_yaxes(showline=True, linewidth=2, linecolor="white", row=2, col=1)

        fig.update_layout(
            template="plotly_dark",
            title="Evolución de Estrategia y Mercado",
            width=1400,
            height=800,
            margin=dict(t=80, b=40, l=40, r=40),
            title_x=0.5
        )

        fig.show()

        
if __name__ == "__main__":
    
    """
    Notas:
        - El TP, SL y el Cost están en términos porcentuales.
        - El leverage 1 por defecto.
    """
    ############################################################################
    ########################### PRUEBA DE ROBUSTEZ #############################
    ############################################################################
    
    ################## BACKTESTING CON DATOS DE ENTRENAMIENTO ##################
    # Datos de 1 hora.
    df_highest_timeframe = pd.read_csv(r'C:\Users\LUIS\Desktop\FrameWork_Concurso\datos\TSLA_1h.csv')
    df_highest_timeframe['Date'] = pd.to_datetime(df_highest_timeframe['Date'])
    df_highest_timeframe.set_index('Date', inplace=True) 

    # Datos de 5 minutos.
    df_lowest_timeframe = pd.read_csv(r'C:\Users\LUIS\Desktop\FrameWork_Concurso\datos\TSLA_5m.csv')
    df_lowest_timeframe['Date'] = pd.to_datetime(df_lowest_timeframe['Date'])
    df_lowest_timeframe.set_index('Date', inplace=True)

    # Parámetros de Inicialización

    df_h_t_train = df_highest_timeframe[(df_highest_timeframe.index >= '2018-01-01 09:30:00') &
                                                (df_highest_timeframe.index < '2020-01-01 09:30:00')]
    df_l_t_train = df_lowest_timeframe[(df_lowest_timeframe.index >= '2018-01-01 09:30:00') &
                                                (df_lowest_timeframe.index < '2020-01-01 09:30:00')]
    
    # Definir parámetros iniciales 
    bb_params = {"longitud": 20, "std_dev": 2.0, "ddof": 0, "columna": "Close"}
    dmi_params = {"suavizado_ADX": 14, "longitud_DMI": 14}
    ft_params = {"longitud": 9}
    TP_SL_params = {"tp": 0.02, "sl":-0.01, "cost": 0.001, "leverage": 1}
    
    # Inicializar la Clase
    est7 = Estrategia7_tp_sl(df_h_t_train, df_l_t_train, BB=bb_params, DMI=dmi_params, FT=ft_params, TP_SL=TP_SL_params)
    print(est7)
    
    # Tratar los Datos de Alta y Baja Frecuencia
    est7.find_timestamp_extremum()
    
    # Calcular indicadores y señales de trading
    resultado = est7.calcular()
    print("Cálculo de los Indicadores:\n", est7.estrategia_calculo)
    print("Tendencia Actual:\n", resultado)
    
    # Run_tp_sl
    df_run = est7.run_tp_sl()
    
    # Backtest de la estrategia
    rendimiento_final = est7.backtest_tp_sl()
    print("Retorno Final Acumulado:\n", rendimiento_final)
    
    mis_datos = est7.data
    
    # Heatmap
    est7.heatmap()
    
    # Simulación de Monte Carlo
    est7.monte_carlo(method="compounded", n_sim=500)
    
    # Ratios de Importancia (No optimizados)
    ratios_no_optimizados = est7.ratios_tp_sl()
    
    # Plot
    est7.plot()
    
    # Optimización de parámetros
    bb_range = [range(9, 50)]
    dmi_range = [range(10, 20), range(10, 20)]
    ft_range = [range(5, 15)]
    
    total_combinaciones = list(product(*bb_range, *dmi_range, *ft_range))
    print("Número total de combinaciones:", len(total_combinaciones))
    
    tiempo_inicio = time.time()
    
    optimizacion = est7.optimizar(bb_range, dmi_range, ft_range, max_iter=1_000)
    
    tiempo_fin = time.time()
    print(f"La optimización tomó {tiempo_fin - tiempo_inicio} segundos")
    print("Optimización:\n", optimizacion)
    
    ##################################################
    # Seleccionar mejores parámetros encontrados
    bb_params = {"longitud": optimizacion["longitud_BB"].iloc[0], "std_dev": 2.0, "ddof": 0, "columna": "Close"}
    dmi_params = {"suavizado_ADX": optimizacion["suavizado_ADX"].iloc[0], "longitud_DMI": optimizacion["longitud_DMI"].iloc[0]}
    ft_params = {"longitud": optimizacion["longitud_FT"].iloc[0]}
    TP_SL_params = {"tp": 0.02, "sl":-0.01, "cost": 0.001, "leverage": 1}

    # Nueva instancia con parámetros optimizados
    est7 = Estrategia7_tp_sl(df_h_t_train, df_l_t_train, BB=bb_params, DMI=dmi_params, FT=ft_params, TP_SL=TP_SL_params)
    print(est7)
    
    # Tratar los Datos de Alta y Baja Frecuencia
    est7.find_timestamp_extremum()
    
    # Calcular indicadores y señales de trading optimizado
    resultado = est7.calcular()
    print("Cálculo de los Indicadores:\n", est7.estrategia_calculo)
    print("Tendencia Actual:\n", resultado)
    
    # Run_tp_sl
    df_run = est7.run_tp_sl()
    
    # Backtest de la estrategia
    rendimiento_final = est7.backtest_tp_sl()
    print("Retorno Final Acumulado:\n", rendimiento_final)
    
    # Heatmap
    est7.heatmap()
    
    # Simulación de Monte Carlo
    est7.monte_carlo(method="compounded", n_sim=500)
    
    # Ratios Optimizados
    ratios_optimizados = est7.ratios_tp_sl()
    
    # Plot
    est7.plot()

    ################## BACKTESTING CON DATOS DE VALIDACIÓN ##################
    # Parámetros de Inicialización

    df_h_t_val = df_highest_timeframe[(df_highest_timeframe.index >= '2020-01-01 09:30:00') &
                                                (df_highest_timeframe.index < '2022-01-01 09:30:00')]
    df_l_t_val = df_lowest_timeframe[(df_lowest_timeframe.index >= '2020-01-01 09:30:00') &
                                                (df_lowest_timeframe.index < '2022-01-01 09:30:00')]
    
    # Nueva instancia con parámetros optimizados
    est7 = Estrategia7_tp_sl(df_h_t_val, df_l_t_val, BB=bb_params, DMI=dmi_params, FT=ft_params, TP_SL=TP_SL_params)
    print(est7)
    
    # Tratar los Datos de Alta y Baja Frecuencia
    est7.find_timestamp_extremum()
    
    # Calcular indicadores y señales de trading optimizado
    resultado = est7.calcular()
    print("Cálculo de los Indicadores:\n", est7.estrategia_calculo)
    print("Tendencia Actual:\n", resultado)
    
    # Run_tp_sl
    df_run = est7.run_tp_sl()
    
    # Backtest de la estrategia
    rendimiento_final = est7.backtest_tp_sl()
    print("Retorno Final Acumulado:\n", rendimiento_final)
    
    # Heatmap
    est7.heatmap()
    
    # Simulación de Monte Carlo
    est7.monte_carlo(method="compounded", n_sim=500)
    
    # Ratios de Validación
    ratios_val = est7.ratios_tp_sl()
    
    # Plot
    est7.plot()
    
    ################## BACKTESTING CON DATOS DE PRUEBA ######################
    # Parámetros de Inicialización

    df_h_t_test = df_highest_timeframe[(df_highest_timeframe.index >= '2022-01-01 09:30:00') &
                                                (df_highest_timeframe.index < '2025-01-01 09:30:00')]
    df_l_t_test = df_lowest_timeframe[(df_lowest_timeframe.index >= '2022-01-01 09:30:00') &
                                                (df_lowest_timeframe.index < '2025-01-01 09:30:00')]
    
    # Nueva instancia con parámetros optimizados
    est7 = Estrategia7_tp_sl(df_h_t_test, df_l_t_test, BB=bb_params, DMI=dmi_params, FT=ft_params, TP_SL=TP_SL_params)
    print(est7)
    
    # Tratar los Datos de Alta y Baja Frecuencia
    est7.find_timestamp_extremum()
    
    # Calcular indicadores y señales de trading optimizado
    resultado = est7.calcular()
    print("Cálculo de los Indicadores:\n", est7.estrategia_calculo)
    print("Tendencia Actual:\n", resultado)
    
    # Run_tp_sl
    df_run = est7.run_tp_sl()
    
    # Backtest de la estrategia
    rendimiento_final = est7.backtest_tp_sl()
    print("Retorno Final Acumulado:\n", rendimiento_final)
    
    # Heatmap
    est7.heatmap()
    
    # Simulación de Monte Carlo
    est7.monte_carlo(method="compounded", n_sim=500)
    
    # Ratios de Prueba
    ratios_prueba = est7.ratios_tp_sl()
    
    # Plot
    est7.plot()

    ################## BACKTESTING ACTUAL #################################
    # Parámetros de Inicialización

    df_h_t_test = df_highest_timeframe[(df_highest_timeframe.index >= '2025-01-01 09:30:00')]
    df_l_t_test = df_lowest_timeframe[(df_lowest_timeframe.index >= '2025-01-01 09:30:00')]
    
    # Nueva instancia con parámetros optimizados
    est7 = Estrategia7_tp_sl(df_h_t_test, df_l_t_test, BB=bb_params, DMI=dmi_params, FT=ft_params, TP_SL=TP_SL_params)
    print(est7)
    
    # Tratar los Datos de Alta y Baja Frecuencia
    est7.find_timestamp_extremum()
    
    # Calcular indicadores y señales de trading optimizado
    resultado = est7.calcular()
    print("Cálculo de los Indicadores:\n", est7.estrategia_calculo)
    print("Tendencia Actual:\n", resultado)
    
    # Run_tp_sl
    df_run = est7.run_tp_sl()
    
    # Backtest de la estrategia
    rendimiento_final = est7.backtest_tp_sl()
    print("Retorno Final Acumulado:\n", rendimiento_final)
    
    # Heatmap
    est7.heatmap()
    
    # Simulación de Monte Carlo
    est7.monte_carlo(method="compounded", n_sim=500)
    
    # Ratios Actuales
    ratios_actuales = est7.ratios_tp_sl()
    
    # Plot
    est7.plot()
    
    
