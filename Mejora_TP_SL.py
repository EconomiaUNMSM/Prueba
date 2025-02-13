# Importar Librerías (adaptación de la estrategia 7)
import yfinance as yf
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

class Estrategia7_tp_sl:
    __version__ = 1.0
    
    def __init__(self, data: pd.DataFrame, df_lowest_timeframe: pd.DataFrame, **kwargs) -> None:
        
        # Atributos
        self.data = data.copy()
        self.df_lowest_timeframe = df_lowest_timeframe.copy()
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
        
        # Establecer nuevas columnas
        self.data["Low_time"] = np.nan
        self.data["High_time"] = np.nan
        self.data["First"] = np.nan
    
        # Realiza un bucle para descubrir cuál de los Take Profit y Stop Loss aparece primero
        for i in tqdm(range(len(self.data) - 1)):
            # Extraer valores del marco de datos del período de tiempo más bajo
            start = self.data.index[i]
            end = self.data.index[i + 1]
            row_lowest_timeframe = self.df_lowest_timeframe.loc[start:end].iloc[:-1]
    
            # Extraer la marca de tiempo del máximo y el mínimo durante el período (marco de tiempo más alto)
            try:
                high = row_lowest_timeframe["High"].idxmax()
                low = row_lowest_timeframe["Low"].idxmin()
    
                self.data.loc[start, "Low_time"] = low
                self.data.loc[start, "High_time"] = high
    
            except Exception as e:
                print(e)
                self.data.loc[start, "Low_time"] = start
                self.data.loc[start, "High_time"] = start
    
        # Descubre cuál aparece primero
        self.data.loc[self.data["High_time"] > self.data["Low_time"], "First"] = 1
        self.data.loc[self.data["High_time"] < self.data["Low_time"], "First"] = 2
        self.data.loc[self.data["High_time"] == self.data["Low_time"], "First"] = 0
    
        # Verificar el número de filas sin TP y SL al mismo tiempo
        percentage_garbage_row = len(self.data.loc[self.data["First"] == 0].dropna()) / len(self.data) * 100
    
        if percentage_garbage_row < 95:
            print(f"WARNINGS: Garbage row: {'%.2f' % percentage_garbage_row} %")
        
        # Transformar las columnas en columnas de fecha y hora
        self.data.High_time = pd.to_datetime(self.data.High_time)
        self.data.Low_time = pd.to_datetime(self.data.Low_time)
        
        # Eliminamos la última fila porque no encontramos el extremo
        self.data = self.data.iloc[:-1]
        
        # Específico de los datos actuales
        if "Date" in self.data.columns:
            del self.data["Date"]
            
            return self.data
          
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
        
        self.direccion_mercado = pd.DataFrame(self.direccion_mercado, columns=["Señal"])
        self.data["Señal"] = self.direccion_mercado
        
        # Inicializar la columna 'duration' si no existe
        if 'duration' not in self.data.columns:
            self.data['duration'] = pd.Timedelta(0)  # Establecer valor predeterminado en 0 (Timedelta)

        buy = False
        sell = False
        self.data["returns"] = np.nan  # Inicializamos las ganancias (si no existen)
        
        for i in range(len(self.data)):

            # Extraer datos de la fila actual
            row = self.data.iloc[i]

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
                        self.data.loc[row.name, "returns"] = (self.TP_SL.get("tp", 0.01) - self.TP_SL.get("cost", 0.001)) * self.TP_SL.get("leverage", 1)
                        # Calcular la duración como la diferencia entre las fechas
                        if self.data.loc[row.name, "duration"] == pd.Timedelta(0):
                            self.data.loc[row.name, "duration"] = (row["Low_time"] - open_buy_date)

                    elif row["First"] == 1:
                        self.data.loc[row.name, "returns"] = (self.TP_SL.get("sl", -0.01) - self.TP_SL.get("cost", 0.001)) * self.TP_SL.get("leverage", 1)
                        # Calcular la duración como la diferencia entre las fechas
                        if self.data.loc[row.name, "duration"] == pd.Timedelta(0):
                            self.data.loc[row.name, "duration"] = (row["Low_time"] - open_buy_date)

                    buy = False
                    open_buy_price = None
                    var_buy_high = 0
                    var_buy_low = 0
                    open_buy_date = None

                elif var_buy_high > self.TP_SL.get("tp", 0.01):
                    self.data.loc[row.name, "returns"] = (self.TP_SL.get("tp", 0.01) - self.TP_SL.get("cost", 0.001)) * self.TP_SL.get("leverage", 1)
                    # Calcular la duración como la diferencia entre las fechas
                    if self.data.loc[row.name, "duration"] == pd.Timedelta(0):
                        self.data.loc[row.name, "duration"] = (row["Low_time"] - open_buy_date)
                    buy = False
                    open_buy_price = None
                    var_buy_high = 0
                    var_buy_low = 0
                    open_buy_date = None
                    
                elif var_buy_low < self.TP_SL.get("sl", -0.01):
                    self.data.loc[row.name, "returns"] = (self.TP_SL.get("sl", -0.01) - self.TP_SL.get("cost", 0.001)) * self.TP_SL.get("leverage", 1)
                    # Calcular la duración como la diferencia entre las fechas
                    if self.data.loc[row.name, "duration"] == pd.Timedelta(0):
                        self.data.loc[row.name, "duration"] = (row["Low_time"] - open_buy_date)
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
                        self.data.loc[row.name, "returns"] = (self.TP_SL.get("tp", 0.01) - self.TP_SL.get("cost", 0.001)) * self.TP_SL.get("leverage", 1)
                        # Calcular la duración como la diferencia entre las fechas
                        if self.data.loc[row.name, "duration"] == pd.Timedelta(0):
                            self.data.loc[row.name, "duration"] = (row["Low_time"] - open_sell_date)

                    elif row["First"] == 2:
                        self.data.loc[row.name, "returns"] = (self.TP_SL.get("sl", -0.01) - self.TP_SL.get("cost", 0.001)) * self.TP_SL.get("leverage", 1)
                        # Calcular la duración como la diferencia entre las fechas
                        if self.data.loc[row.name, "duration"] == pd.Timedelta(0):
                            self.data.loc[row.name, "duration"] = (row["Low_time"] - open_sell_date)

                    sell = False
                    open_sell_price = None
                    var_sell_high = 0
                    var_sell_low = 0
                    open_sell_date = None

                elif var_sell_low > self.TP_SL.get("tp", 0.01):
                    self.data.loc[row.name, "returns"] = (self.TP_SL.get("tp", 0.01) - self.TP_SL.get("cost", 0.001)) * self.TP_SL.get("leverage", 1)
                    # Calcular la duración como la diferencia entre las fechas
                    if self.data.loc[row.name, "duration"] == pd.Timedelta(0):
                        self.data.loc[row.name, "duration"] = (row["Low_time"] - open_sell_date)
                    sell = False
                    open_sell_price = None
                    var_sell_high = 0
                    var_sell_low = 0
                    open_sell_date = None

                elif var_sell_high < self.TP_SL.get("sl", -0.01):
                    self.data.loc[row.name, "returns"] = (self.TP_SL.get("sl", -0.01) - self.TP_SL.get("cost", 0.001)) * self.TP_SL.get("leverage", 1)
                    # Calcular la duración como la diferencia entre las fechas
                    if self.data.loc[row.name, "duration"] == pd.Timedelta(0):
                        self.data.loc[row.name, "duration"] = (row["Low_time"] - open_sell_date)
                    sell = False
                    open_sell_price = None
                    var_sell_high = 0
                    var_sell_low = 0
                    open_sell_date = None

        # Rellenar con 0 las filas sin retorno
        self.data["returns"] = self.data["returns"].fillna(value=0)
        data = self.data
        
        return data
    
    def backtest_tp_sl(self):
        if 'returns' not in self.data.columns:
            raise RuntimeError("Ejecutar el método de .run_tp_sl() antes de correr el backtest")
            
        # Calcular los Retornos Diarios
        portfolio = self.data["returns"]
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
        self.backtest_tp_sl()
        
        return resultados
    
    def ratios(self, ben: str="^GSPC", timeframe: int=252):
        
        # Cálculo del TRADE LIFETIME (tiempo de vida)
        sum_dates = self.data["duration"]
        seconds = np.round(np.mean(list(sum_dates.loc[sum_dates != 0])).total_seconds())
        minutes = seconds // 60
        minutes_left = int(minutes % 60)
        hours = int(minutes // 60)
        
        # Cálculo de los Retornos Diarios
        portfolio = self.data["returns"]
        portfolio = portfolio.reset_index(drop=False)  # Reseteamos para agrupar correctamente
        portfolio["Time"] = portfolio["Date"].dt.date  # Creamos una nueva columna de fecha
        portfolio = portfolio.groupby("Time")["returns"].sum()  # Agrupamos por fecha

        ######################### Cálculo del Beta ##################################
        # Importation of benchmark
        benchmark = yf.download(ben)["Adj Close"].pct_change(1).dropna()

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
        num_trades = len(self.data[self.data["Señal"] == 1]) + len(self.data[self.data["Señal"] == -1])
        # Ratio Long/ Short
        long_trades = len(self.data[self.data["Señal"] == 1])
        short_trades = len(self.data[self.data["Señal"] == -1])
        long_short_ratio = long_trades / short_trades if short_trades > 0 else np.nan
        # Hit Ratio
        winning_trades = len(self.data[self.data["returns"] > 0])
        hit_ratio = winning_trades / num_trades if num_trades > 0 else 0
        # Profit Factor
        gross_profit = self.data[self.data["returns"] > 0]["returns"].sum()
        gross_loss = abs(self.data[self.data["returns"] < 0]["returns"].sum())
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
        
        return calculo_ratios
    
    def plot(self) -> None:
        
        # Graficar
        fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(18, 14), gridspec_kw={"height_ratios": [1, 4, 1, 1, 1]},
                                 dpi=300)
        
        # Subplot 1: Rendimiento Acumulado
        axes[0].plot(self.rendimiento_final_estrategia.fillna(value=1), label="Rendimiento Acumulado")
        axes[0].set_title(f"Rendimiento Acumulado (Periodo {self.data.index[0].strftime('%Y/%m/%d')} - {self.data.index[-1].strftime('%Y/%m/%d')})",
                          fontsize=18, fontweight="bold", loc="left")
        axes[0].legend()
        axes[0].grid()
        axes[0].set_facecolor("#F7F7F7")
        
        # Subplot 2: Gráfico de Velas        
        axes[1].set_title("Gráfico de Velas", loc="center", fontsize=18, fontweight="bold")
        mpf.plot(self.data, type="candle", style="yahoo", ax=axes[1])
        min_ytick, max_ytick = axes[1].get_yticks().min(), axes[1].get_yticks().max()
        yticks = np.linspace(start=min_ytick, stop=max_ytick, num=15)
        axes[1].set_yticks(yticks)
        axes[1].grid()
        axes[1].set_ylim([self.data["Low"].min() * 0.99, self.data["High"].max() * 1.01])
        axes[1].set_xticklabels(self.data.index.strftime("%Y/%m/%d"), rotation=0)
        axes[1].set_facecolor("#F7F7F7")
        
        # Subplot 2: Bandas de Bollinger
        axes[2].plot(self.estrategia_calculo["BB"], label=["Media Móvil", "Banda Superior", "Banda Inferior"])
        axes[2].plot(self.data["Close"], label="Precios de Cierre")
        axes[2].set_title("Bandas de Bollinger", fontsize=18, loc="left", weight="bold")
        axes[2].legend(loc="lower left")
        axes[2].grid()
        
        # Subplot 3: DMI
        axes[3].plot(self.estrategia_calculo["DMI"], label=["+DI", "MA", "-DI"])
        axes[3].legend(fontsize=15, loc="upper right")
        axes[3].set_title("Índice de Movimiento Direccional (+DI, -DI y ADX)", size=20, fontweight="bold")
        axes[3].grid(True)
        
        # Subplot 4: Fisher Transform
        axes[4].plot(self.estrategia_calculo["FT"])
        axes[4].axhline(y=0, label="Neutral", color="gray", lw=2, linestyle="--")
        axes[4].set_title("Indicador de Fisher Transform", fontsize=18, loc="left", weight="bold")
        axes[4].legend(loc="lower left")
        axes[4].grid()
        
        # Grosor el marco
        for ax in axes:
            ax.spines["top"].set_linewidth(2.5)
            ax.spines["right"].set_linewidth(2.5)
            ax.spines["bottom"].set_linewidth(2.5)
            ax.spines["left"].set_linewidth(2.5)
            
        # Estilos generales del gráfico
        plt.tight_layout()
        plt.show()

        
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
    df_highest_timeframe = pd.read_csv(r'.\TSLA_1h.csv')
    df_highest_timeframe['Date'] = pd.to_datetime(df_highest_timeframe['Date'])
    df_highest_timeframe.set_index('Date', inplace=True) 

    # Datos de 5 minutos.
    df_lowest_timeframe = pd.read_csv(r'.\TSLA_5m.csv')
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
    TP_SL_params = {"tp": 0.01, "sl":-0.01, "cost": 0, "leverage": 1}
    
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
    
    # Ratios de Importancia (No optimizados)
    ratios_no_optimizados = est7.ratios()
    
    # Plot
    est7.plot()
    
    # Optimización de parámetros
    bb_range = [range(9, 50)]
    dmi_range = [range(10, 20), range(10, 20)]
    ft_range = [range(5, 15)]
    
    total_combinaciones = list(product(*bb_range, *dmi_range, *ft_range))
    print("Número total de combinaciones:", len(total_combinaciones))
    
    tiempo_inicio = time.time()
    
    optimizacion = est7.optimizar(bb_range, dmi_range, ft_range, max_iter=10)
    
    tiempo_fin = time.time()
    print(f"La optimización tomó {tiempo_fin - tiempo_inicio} segundos")
    print("Optimización:\n", optimizacion)

    

    
    
    
    
    ##################################################
    # Seleccionar mejores parámetros encontrados
    bb_params = {"longitud": optimizacion["longitud_BB"].iloc[0], "std_dev": 2.0, "ddof": 0, "columna": "Close"}
    dmi_params = {"suavizado_ADX": optimizacion["suavizado_ADX"].iloc[0], "longitud_DMI": optimizacion["longitud_DMI"].iloc[0]}
    ft_params = {"longitud": optimizacion["longitud_FT"].iloc[0]}
    TP_SL_params = {"tp": 0.02, "sl":-0.01, "cost": 0, "leverage": 1}

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
    
    # Ratios Optimizados
    ratios_optimizados = est7.ratios()
    
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

    # Ratios de Validación
    ratios_val = est7.ratios()
    
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
    
    # Ratios de Prueba
    ratios_prueba = est7.ratios()
    
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
    
    # Ratios Actuales
    ratios_actuales = est7.ratios()
    
    # Plot
    est7.plot()
    
    



    
    
        
        

























