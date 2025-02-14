# Importar Librerías
import pandas as pd
import threading
import time
from ibapi.contract import Contract
from ibapi.client import EClient
from ibapi.wrapper import EWrapper

# Clase para obtener datos históricos
class IB_Datos_Instrumentos(EClient, EWrapper):
    def __init__(self):
        EClient.__init__(self, self)
        self.datos = []
        self.event = threading.Event()  # Para esperar la respuesta de IB
        
    def historicalData(self, reqId, bar):
        self.datos.append({
            "Date": bar.date, "Open": bar.open, "High": bar.high,
            "Low": bar.low, "Close": bar.close, "Volume": bar.volume
        })

    def historicalDataEnd(self, reqId, start, end):
        print(f"✅ Datos recibidos hasta: {end}")
        self.event.set()  # Marca que la solicitud ha terminado

# Configurar contrato
contrato = Contract()
contrato.symbol = "AAPL"     # Ticker de la Empresa
contrato.secType = "STK"     # STK (accion), OPT (Opciones Financieras), FUT (Futuros).
contrato.exchange = "SMART"  # Buscador Inteligente de IBKR. Si se poporciona un mercado específico, necesitará una cuenta de corretaje.
contrato.currency = "USD"    # Moneda.

# Conectar con IB
IB_conexion = IB_Datos_Instrumentos()
IB_conexion.connect(host="127.0.0.1", port=7497, clientId=1)
threading.Thread(target=IB_conexion.run).start()
time.sleep(1)

# Solicitud de Datos Disponibles.

def extraer_datos_ibkr_1d(years, IB_conexion, contrato):
    """
    Extrae datos históricos diarios (1 día por barra) para un contrato dado usando IBKR.
    
    Parámetros:
      - years (int): Número de años de datos a extraer.
      - IB_conexion: Objeto de conexión a IB.
      - contrato: Objeto de contrato configurado.

    Retorna:
      - df_final (DataFrame) con las columnas: Open, High, Low, Close, Volume.
    """
    
    total_iter = years
    end_date = ""
    dataframes = []
    errores_seguidos = 0

    for i in range(total_iter):
        print(f"\n📡 Solicitud {i+1}/{total_iter} → End Date: {end_date or 'Actual'}")
        
        IB_conexion.datos.clear()
        IB_conexion.event.clear()
        
        IB_conexion.reqHistoricalData(
            reqId=1,
            contract=contrato,
            endDateTime=end_date,
            durationStr="1 Y",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=1,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )
        
        IB_conexion.event.wait(timeout=10)
        
        if not IB_conexion.datos:
            print("⚠️ No se recibieron datos, terminando intento...")
            errores_seguidos += 1
            if errores_seguidos >= 3:
                print("❌ Demasiados errores seguidos. Deteniendo el proceso.")
                break
            continue
        
        df_temp = pd.DataFrame(IB_conexion.datos)
        
        if not df_temp.empty and (len(dataframes) == 0 or df_temp["Date"].max() < dataframes[-1]["Date"].min()):
            dataframes.append(df_temp)
            print(f"📊 Datos agregados hasta: {df_temp['Date'].min()}")
        
        end_date = pd.to_datetime(df_temp["Date"].min()).strftime("%Y%m%d %H:%M:%S")
        errores_seguidos = 0
        time.sleep(1)
    
    if dataframes:
        df_final = pd.concat(dataframes).drop_duplicates().reset_index(drop=True)

        # Convertir 'Date' a datetime y establecer como índice
        df_final.rename(columns={'Date': 'Date'}, inplace=True)  # Asegurar que la columna se llama 'Date'
        df_final['Date'] = pd.to_datetime(df_final['Date'])
        df_final.set_index('Date', inplace=True)

        # Mantener solo las columnas deseadas
        columnas_deseadas = ['Open', 'High', 'Low', 'Close', 'Volume']
        df_final = df_final[columnas_deseadas]

        # Ordenar de menor a mayor fecha
        df_final.sort_index(inplace=True)

        # Guardar en CSV manteniendo el índice
        df_final.to_csv(f"{contrato.symbol}_1d.csv")

        print(f"\n✅ Datos almacenados correctamente en {contrato.symbol}_1d.csv")
        print(df_final.head())

        return df_final
    else:
        print("❌ No se descargaron datos.")
        return None

def extraer_datos_ibkr_1h(years, IB_conexion, contrato):
    """
    Extrae datos históricos de 1 hora para un contrato dado usando IBKR.
    
    La función realiza una iteración por cada año a extraer (según el parámetro 'years')
    y utiliza la función reqHistoricalData de la conexión IB (IB_conexion). Cada iteración
    descarga datos del período de 1 año, actualizando el endDate para retroceder en el tiempo.
    
    Parámetros:
      - years (int): Número de años de datos a extraer.
      - IB_conexion: Objeto de conexión a IB que debe tener los atributos 'datos' (lista) y 
                     'event' (un threading.Event()) y el método reqHistoricalData().
      - contrato: Objeto de contrato configurado (por ejemplo, con atributos symbol, secType, exchange, currency).
    
    Retorna:
      - df_final (DataFrame): DataFrame con índice "Date" y columnas: Open, High, Low, Close, Volume,
        o None si no se pudieron obtener datos.
    """
    
    # Variables iniciales
    end_date = ""  # Al estar vacío, IB descarga desde la fecha actual
    total_iter = years  # Se realizará una iteración por cada año
    dataframes = []
    errores_seguidos = 0  # Contador de intentos fallidos
    
    # Bucle para descargar datos en fragmentos de 1 año hacia atrás
    for i in range(total_iter):
        print(f"\n📡 Solicitud {i+1}/{total_iter} → End Date: {end_date or 'Actual'}")
        
        # Limpiar datos previos y resetear evento
        IB_conexion.datos.clear()
        IB_conexion.event.clear()
        
        # Solicitar datos históricos de 1 año, con velas de 1 hora
        IB_conexion.reqHistoricalData(
            reqId=1,
            contract=contrato,
            endDateTime=end_date,
            durationStr="1 Y",
            barSizeSetting="1 hour",
            whatToShow="TRADES",
            useRTH=1,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )
        
        # Esperar la respuesta (hasta 10 segundos)
        IB_conexion.event.wait(timeout=10)
        
        # Si no se recibieron datos, contabilizar error y, en caso de 3 fallos consecutivos, salir
        if not IB_conexion.datos:
            print("⚠️ No se recibieron datos, terminando intento...")
            errores_seguidos += 1
            if errores_seguidos >= 3:
                print("❌ Demasiados errores seguidos. Deteniendo el proceso.")
                break
            continue
        
        # Convertir los datos recibidos a DataFrame
        df_temp = pd.DataFrame(IB_conexion.datos)
        
        # Si el DataFrame no está vacío y no es duplicado, lo agregamos
        if not df_temp.empty and (len(dataframes) == 0 or df_temp["Date"].max() < dataframes[-1]["Date"].min()):
            dataframes.append(df_temp)
            print(f"📊 Datos agregados hasta: {df_temp['Date'].min()}")
        
        # Actualizar end_date para la siguiente iteración (se usa la fecha mínima del DataFrame)
        end_date = pd.to_datetime(df_temp["Date"].min()).strftime("%Y%m%d %H:%M:%S")
        
        # Resetear contador de errores y esperar un segundo para evitar bloqueos
        errores_seguidos = 0
        time.sleep(1)
    
    if dataframes:
        # Unir todos los DataFrames descargados y eliminar duplicados
        df_final = pd.concat(dataframes).drop_duplicates().reset_index(drop=True)
        
        # Convertir 'Date' a datetime, ordenar y establecerla como índice
        df_final['Date'] = pd.to_datetime(df_final['Date'])
        df_final.sort_values(by='Date', inplace=True)
        df_final.set_index('Date', inplace=True)
        
        # Mantener solo las columnas deseadas en el orden correcto
        columnas_deseadas = ['Open', 'High', 'Low', 'Close', 'Volume']
        df_final = df_final[columnas_deseadas]
        
        # Guardar en CSV manteniendo el índice (que es 'Date')
        df_final.to_csv(f"{contrato.symbol}_1h.csv")
        print(f"\n✅ Datos almacenados correctamente en {contrato.symbol}_1h.csv")
        print(df_final.head())
        
        return df_final
    else:
        print("❌ No se descargaron datos.")
        return None   

def extraer_datos_ibkr_5m(years, IB_conexion, contrato):
    """
    Extrae datos históricos de 5 minutos para un contrato dado usando IBKR.
    
    La función realiza iteraciones mensuales para cubrir el rango total de 'years' años.
    Cada iteración descarga datos de 1 mes (durationStr="1 M") con velas de 5 minutos
    (barSizeSetting="5 mins"). Se utiliza la lógica de actualización de la fecha final
    (end_date) para avanzar hacia atrás en el tiempo, evitando duplicados y controlando errores.
    
    Parámetros:
      - years (int): Número de años de datos a extraer.
      - IB_conexion: Objeto de conexión a IB que debe tener los atributos 'datos' (lista) y 
                     'event' (un threading.Event()) y el método reqHistoricalData().
      - contrato: Objeto de contrato configurado (por ejemplo, con atributos symbol, secType, exchange, currency).
    
    Retorna:
      - df_final (DataFrame): DataFrame con índice "Date" y columnas: Open, High, Low, Close, Volume,
        o None si no se pudieron obtener datos.
    """
    
    total_iter = years * 12   # Iteraciones mensuales (por ejemplo, 10 años → 120 iteraciones)
    end_date = ""             # Vacío: IB descarga hasta la fecha actual
    dataframes = []
    errores_seguidos = 0
    
    for i in range(total_iter):
        print(f"\n📡 Solicitud {i+1}/{total_iter} → End Date: {end_date or 'Actual'}")
        
        # Limpiar datos previos y resetear el evento
        IB_conexion.datos.clear()
        IB_conexion.event.clear()
        
        # Solicitar datos históricos de 1 mes con velas de 5 minutos
        IB_conexion.reqHistoricalData(
            reqId=1,
            contract=contrato,
            endDateTime=end_date,
            durationStr="1 M",          # 1 Mes por solicitud
            barSizeSetting="5 mins",    # Intervalos de 5 minutos
            whatToShow="TRADES",
            useRTH=1,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )
        
        # Esperar la respuesta (hasta 10 segundos)
        IB_conexion.event.wait(timeout=10)
        
        if not IB_conexion.datos:
            print("⚠️ No se recibieron datos, terminando intento...")
            errores_seguidos += 1
            if errores_seguidos >= 3:
                print("❌ Demasiados errores seguidos. Deteniendo el proceso.")
                break
            continue
        
        # Convertir los datos recibidos a un DataFrame
        df_temp = pd.DataFrame(IB_conexion.datos)
        
        # Si el DataFrame no está vacío y no es duplicado, lo agregamos
        if not df_temp.empty and (len(dataframes) == 0 or df_temp["Date"].max() < dataframes[-1]["Date"].min()):
            dataframes.append(df_temp)
            print(f"📊 Datos agregados hasta: {df_temp['Date'].min()}")
        
        # Actualizar end_date para la siguiente iteración (se usa la fecha mínima del fragmento actual)
        end_date = pd.to_datetime(df_temp["Date"].min()).strftime("%Y%m%d %H:%M:%S")
        errores_seguidos = 0
        time.sleep(1)
    
    if dataframes:
        # Unir todos los fragmentos y eliminar duplicados
        df_final = pd.concat(dataframes).drop_duplicates().reset_index(drop=True)
        
        # Convertir 'Date' a datetime, ordenar y establecer como índice
        df_final['Date'] = pd.to_datetime(df_final['Date'])
        df_final.sort_values(by='Date', inplace=True)
        df_final.set_index('Date', inplace=True)
        
        # Mantener solo las columnas deseadas en el orden correcto
        columnas_deseadas = ['Open', 'High', 'Low', 'Close', 'Volume']
        df_final = df_final[columnas_deseadas]
        
        # Guardar en CSV manteniendo el índice (el índice se guarda como la columna 'Date') 
        df_final.to_csv(f"{contrato.symbol}_5m.csv", index=True)
        print(f"\n✅ Datos almacenados correctamente en {contrato.symbol}_5m.csv")
        print(df_final.head())
        
        return df_final
    else:
        print("❌ No se descargaron datos.")
        return None

def extraer_datos_ibkr_1m(years, IB_conexion, contrato):
    
    """
    Extrae datos históricos de 1 minuto para un contrato dado usando IBKR.
    
    La función realiza iteraciones diarias (1 D) para cubrir el rango total de 'years' años.
    Cada iteración descarga datos de 1 día con barras de 1 minuto. Se utiliza la lógica de
    actualización de la fecha final (end_date) para retroceder en el tiempo, evitando duplicados
    y controlando errores.
    
    Parámetros:
      - years (int): Número de años de datos a extraer.
      - IB_conexion: Objeto de conexión a IB que debe tener los atributos 'datos' (lista),
                     'event' (un threading.Event()) y el método reqHistoricalData().
      - contrato: Objeto de contrato configurado (con atributos como symbol, secType, exchange, currency).
    
    Retorna:
      - df_final (DataFrame): DataFrame con índice "Date" y columnas: Open, High, Low, Close, Volume,
        o None si no se pudieron obtener datos.
    """
    total_iter = years * 365  
    end_date = ""  # Al estar vacío, IB descarga desde la fecha actual
    dataframes = []
    errores_seguidos = 0  # Contador de intentos fallidos
    
    # Bucle para descargar datos diarios hacia atrás
    for i in range(total_iter):
        print(f"\n📡 Solicitud {i+1}/{total_iter} → End Date: {end_date or 'Actual'}")
        
        # Limpiar datos previos y resetear el evento
        IB_conexion.datos.clear()
        IB_conexion.event.clear()
        
        # Solicitar datos históricos: 1 día de datos con velas de 1 minuto
        IB_conexion.reqHistoricalData(
            reqId=1,
            contract=contrato,
            endDateTime=end_date,
            durationStr="1 D",        # 1 día de datos
            barSizeSetting="1 min",   # Barras de 1 minuto
            whatToShow="TRADES",
            useRTH=1,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )
        
        # Esperar la respuesta (hasta 10 segundos)
        IB_conexion.event.wait(timeout=10)
        
        if not IB_conexion.datos:
            print("⚠️ No se recibieron datos, terminando intento...")
            errores_seguidos += 1
            if errores_seguidos >= 3:
                print("❌ Demasiados errores seguidos. Deteniendo el proceso.")
                break
            continue
        
        # Convertir los datos recibidos a DataFrame
        df_temp = pd.DataFrame(IB_conexion.datos)
        
        # Agregar el DataFrame si no es duplicado (comparando la fecha máxima del fragmento con la del último agregado)
        if not df_temp.empty and (len(dataframes) == 0 or df_temp["Date"].max() < dataframes[-1]["Date"].min()):
            dataframes.append(df_temp)
            print(f"📊 Datos agregados hasta: {df_temp['Date'].min()}")
        
        # Actualizar end_date para la siguiente iteración (se usa la fecha mínima del fragmento actual)
        end_date = pd.to_datetime(df_temp["Date"].min()).strftime("%Y%m%d %H:%M:%S")
        
        errores_seguidos = 0
        time.sleep(1)
    
    if dataframes:
        # Unir todos los DataFrames descargados y eliminar duplicados
        df_final = pd.concat(dataframes).drop_duplicates().reset_index(drop=True)
        
        # Convertir 'Date' a datetime, ordenar y establecerla como índice
        df_final['Date'] = pd.to_datetime(df_final['Date'])
        df_final.sort_values(by='Date', inplace=True)
        df_final.set_index('Date', inplace=True)
        
        # Mantener solo las columnas deseadas en el orden correcto
        columnas_deseadas = ['Open', 'High', 'Low', 'Close', 'Volume']
        df_final = df_final[columnas_deseadas]
        
        # Guardar en CSV manteniendo el índice (el índice se guardará como "Date")
        df_final.to_csv(f"{contrato.symbol}_1m.csv", index=True)
        print(f"\n✅ Datos almacenados correctamente en {contrato.symbol}_1m.csv")
        print(df_final.head())
        
        return df_final
    else:
        print("❌ No se descargaron datos.")
        return None    
    
    
if __name__ == "__main__":
    # Descarga de Datos de 1 día para los últimos 10 años.
    extraer_datos_ibkr_1d(10, IB_conexion, contrato)
    df_1d = pd.read_csv(r'.\AAPL_1d.csv')

    extraer_datos_ibkr_1h(10, IB_conexion, contrato)
    df_1h = pd.read_csv(r'.\AAPL_1h.csv')
    
    extraer_datos_ibkr_5m(10, IB_conexion, contrato)
    df_5m = pd.read_csv(r'.\AAPL_5m.csv')
    
    extraer_datos_ibkr_1m(10, IB_conexion, contrato)
    df_1h = pd.read_csv(r'.\AAPL_1m.csv')

    """
    Observaciones:
        - Necesario tener instalado y abierto TraderWorkStation.
        - Configurar en "Config."-> "API" -> "Solo lectura" (remover el visto, para poderse conectar con la API).
        - Si trabaja con IB GateWay, editar la línea 36, en port=7497 -> port=4002
        - Los datos de menor temporalidad tardan mucho en descargarse.
        - Los datos generalmente presentan errores de descarga debido a la masiva solicitud de datos.
        - En la API de IBKR se pueden solicitar las siguientes temporalidades intradía:
            * "1 sec"
            * "5 secs"
            * "15 secs"
            * "1 min"
            * "2 mins"
            * "3 mins"
            * "5 mins"
            * "15 mins"
            * "30 mins"
            * "1 hour"
        - Es cuestión de modificar el archivo según la necesidad (tarea para la casa).
        - Descargas con retraso de 15 minutos (reqMktDataType = 3, datos sin suscripción o cuenta DEMO).
        - Para la solicitud de otro tipo de activos, es necesario nuevos parámetros (investigue).
    
    Usos: 
        - Probar estrategias de Trading Algorítmico.
        - Superada la limitación de descarga de datos de yfinance.
        - Estudios de investigación.
        
    """

    
    
    
    
    