# Importar Librerías
import os
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
    Extrae datos históricos diarios (1 día por barra) para un contrato usando IBKR.
    Si el archivo CSV ya existe, agrega nuevos datos sin duplicar.

    Parámetros:
      - years (int): Años de datos a extraer.
      - IB_conexion: Objeto de conexión a IBKR.
      - contrato: Objeto de contrato configurado.

    Retorna:
      - DataFrame con columnas: Open, High, Low, Close, Volume.
    """

    total_iter = years
    end_date = ""
    dataframes = []
    errores_seguidos = 0
    archivo_csv = f"{contrato.symbol}_1d.csv"

    # Cargar datos previos si el archivo existe
    if os.path.exists(archivo_csv):
        df_existente = pd.read_csv(archivo_csv, parse_dates=["Date"], index_col="Date")
        print(f"📂 Archivo existente encontrado: {archivo_csv} ({len(df_existente)} filas)")
    else:
        df_existente = pd.DataFrame()

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
        
        if not df_temp.empty:
            df_temp["Date"] = pd.to_datetime(df_temp["Date"])
            dataframes.append(df_temp)
            print(f"📊 Datos agregados hasta: {df_temp['Date'].min()}")

        end_date = df_temp["Date"].min().strftime("%Y%m%d %H:%M:%S")
        errores_seguidos = 0
        time.sleep(1)

    if dataframes:
        df_nuevo = pd.concat(dataframes).drop_duplicates().reset_index(drop=True)
        df_nuevo.set_index("Date", inplace=True)

        columnas_deseadas = ["Open", "High", "Low", "Close", "Volume"]
        df_nuevo = df_nuevo[columnas_deseadas]

        # Si ya existe el archivo, fusionar datos y sobrescribir solo registros duplicados
        if not df_existente.empty:
            df_final = pd.concat([df_existente, df_nuevo]).drop_duplicates().sort_index()
        else:
            df_final = df_nuevo

        df_final.to_csv(archivo_csv, index=True)
        print(f"\n✅ Datos actualizados y guardados en {archivo_csv}")
        print(df_final.head())

        return df_final
    else:
        print("❌ No se descargaron datos.")
        return None

def extraer_datos_ibkr_1h(years, IB_conexion, contrato):
    """
    Extrae datos históricos de 1 hora para un contrato usando IBKR y los guarda en CSV.
    Si el archivo ya existe, agrega los datos nuevos y elimina duplicados basándose en la fecha.
    """
    end_date = ""  # IB descarga desde la fecha actual si está vacío
    total_iter = years  # Iteraciones de 1 año cada una
    dataframes = []
    errores_seguidos = 0
    archivo_csv = f"{contrato.symbol}_1h.csv"

    # Cargar datos previos si el archivo existe
    if os.path.exists(archivo_csv):
        df_existente = pd.read_csv(archivo_csv, parse_dates=["Date"], index_col="Date")
        print(f"📂 Archivo existente encontrado: {archivo_csv} ({len(df_existente)} filas)")
    else:
        df_existente = pd.DataFrame()

    for i in range(total_iter):
        print(f"\n📡 Solicitud {i+1}/{total_iter} → End Date: {end_date or 'Actual'}")
        
        IB_conexion.datos.clear()
        IB_conexion.event.clear()
        
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
        
        IB_conexion.event.wait(timeout=10)
        
        if not IB_conexion.datos:
            print("⚠️ No se recibieron datos, terminando intento...")
            errores_seguidos += 1
            if errores_seguidos >= 3:
                print("❌ Demasiados errores seguidos. Deteniendo el proceso.")
                break
            continue
        
        df_temp = pd.DataFrame(IB_conexion.datos)
        
        if not df_temp.empty:
            df_temp["Date"] = pd.to_datetime(df_temp["Date"])
            dataframes.append(df_temp)
            print(f"📊 Datos agregados hasta: {df_temp['Date'].min()}")
        
        end_date = df_temp["Date"].min().strftime("%Y%m%d %H:%M:%S")
        errores_seguidos = 0
        time.sleep(1)

    if dataframes:
        df_nuevo = pd.concat(dataframes).drop_duplicates().reset_index(drop=True)
        df_nuevo.set_index("Date", inplace=True)
        
        columnas_deseadas = ["Open", "High", "Low", "Close", "Volume"]
        df_nuevo = df_nuevo[columnas_deseadas]

        # Si ya existe el archivo, fusionar datos y sobrescribir solo registros duplicados
        if not df_existente.empty:
            df_final = pd.concat([df_existente, df_nuevo]).drop_duplicates().sort_index()
        else:
            df_final = df_nuevo

        df_final.to_csv(archivo_csv, index=True)
        print(f"\n✅ Datos actualizados y guardados en {archivo_csv}")
        print(df_final.head())

        return df_final
    else:
        print("❌ No se descargaron datos.")
        return None

def extraer_datos_ibkr_5m(years, IB_conexion, contrato):
    """
    Extrae datos históricos de 5 minutos para un contrato usando IBKR y los guarda en CSV.
    Si el archivo ya existe, agrega los datos nuevos y elimina duplicados basándose en la fecha.
    """
    total_iter = years * 12  # Iteraciones mensuales
    end_date = ""  # Vacío: IB descarga hasta la fecha actual
    dataframes = []
    errores_seguidos = 0
    archivo_csv = f"{contrato.symbol}_5m.csv"

    # Cargar datos previos si el archivo existe
    if os.path.exists(archivo_csv):
        df_existente = pd.read_csv(archivo_csv, parse_dates=["Date"], index_col="Date")
        print(f"📂 Archivo existente encontrado: {archivo_csv} ({len(df_existente)} filas)")
    else:
        df_existente = pd.DataFrame()

    for i in range(total_iter):
        print(f"\n📡 Solicitud {i+1}/{total_iter} → End Date: {end_date or 'Actual'}")
        
        IB_conexion.datos.clear()
        IB_conexion.event.clear()
        
        IB_conexion.reqHistoricalData(
            reqId=1,
            contract=contrato,
            endDateTime=end_date,
            durationStr="1 M",
            barSizeSetting="5 mins",
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
        
        if not df_temp.empty:
            df_temp["Date"] = pd.to_datetime(df_temp["Date"])
            dataframes.append(df_temp)
            print(f"📊 Datos agregados hasta: {df_temp['Date'].min()}")
        
        end_date = df_temp["Date"].min().strftime("%Y%m%d %H:%M:%S")
        errores_seguidos = 0
        time.sleep(1)

    if dataframes:
        df_nuevo = pd.concat(dataframes).drop_duplicates().reset_index(drop=True)
        df_nuevo.set_index("Date", inplace=True)
        
        columnas_deseadas = ["Open", "High", "Low", "Close", "Volume"]
        df_nuevo = df_nuevo[columnas_deseadas]

        # Si ya existe el archivo, fusionar datos y sobrescribir solo registros duplicados
        if not df_existente.empty:
            df_final = pd.concat([df_existente, df_nuevo]).drop_duplicates().sort_index()
        else:
            df_final = df_nuevo

        df_final.to_csv(archivo_csv, index=True)
        print(f"\n✅ Datos actualizados y guardados en {archivo_csv}")
        print(df_final.head())

        return df_final
    else:
        print("❌ No se descargaron datos.")
        return None

def extraer_datos_ibkr_1m(days, IB_conexion, contrato):
    """
    Extrae datos históricos de 1 minuto para un contrato dado usando IBKR y los guarda en CSV.
    Si el CSV ya existe, apendiza la nueva información y elimina duplicados en base a la fecha.
    """
    total_iter = days   
    end_date = ""  # Al estar vacío, IB descarga desde la fecha actual
    dataframes = []
    errores_seguidos = 0  # Contador de intentos fallidos
    archivo_csv = f"{contrato.symbol}_1m.csv"
    
    # Si el archivo ya existe, cargar datos previos
    if os.path.exists(archivo_csv):
        df_existente = pd.read_csv(archivo_csv, parse_dates=["Date"], index_col="Date")
        print(f"📂 Archivo existente encontrado: {archivo_csv} ({len(df_existente)} filas)")
    else:
        df_existente = pd.DataFrame()  # Crear DataFrame vacío
    
    # Bucle para descargar datos diarios hacia atrás
    for i in range(total_iter):
        print(f"\n📡 Solicitud {i+1}/{total_iter} → End Date: {end_date or 'Actual'}")
        
        IB_conexion.datos.clear()
        IB_conexion.event.clear()
        
        IB_conexion.reqHistoricalData(
            reqId=1,
            contract=contrato,
            endDateTime=end_date,
            durationStr="1 D",
            barSizeSetting="1 min",
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
            if errores_seguidos >= 5:
                print("❌ Demasiados errores seguidos. Deteniendo el proceso.")
                break
            continue
        
        df_temp = pd.DataFrame(IB_conexion.datos)
        
        if not df_temp.empty:
            df_temp["Date"] = pd.to_datetime(df_temp["Date"])
            dataframes.append(df_temp)
            print(f"📊 Datos agregados hasta: {df_temp['Date'].min()}")
        
        end_date = df_temp["Date"].min().strftime("%Y%m%d %H:%M:%S")
        
        errores_seguidos = 0
        time.sleep(1)
    
    if dataframes:
        df_nuevo = pd.concat(dataframes).drop_duplicates().reset_index(drop=True)
        df_nuevo.set_index("Date", inplace=True)
        
        columnas_deseadas = ["Open", "High", "Low", "Close", "Volume"]
        df_nuevo = df_nuevo[columnas_deseadas]

        # Si ya existe el archivo, fusionar datos y sobrescribir solo los registros duplicados
        if not df_existente.empty:
            df_final = pd.concat([df_existente, df_nuevo]).drop_duplicates().sort_index()
        else:
            df_final = df_nuevo

        # Guardar el archivo CSV actualizado
        df_final.to_csv(archivo_csv, index=True)
        print(f"\n✅ Datos actualizados y guardados en {archivo_csv}")
        print(df_final.head())
        
        return df_final
    else:
        print("❌ No se descargaron datos.")
        return None
    
    
if __name__ == "__main__":
    extraer_datos_ibkr_1d(10, IB_conexion, contrato)
    df_1d = pd.read_csv(r'.\AAPL_1d.csv')
    
    extraer_datos_ibkr_1h(1, IB_conexion, contrato)
    df_1h = pd.read_csv(r'.\AAPL_1h.csv')
    
    extraer_datos_ibkr_5m(1, IB_conexion, contrato)
    df_5m = pd.read_csv(r'.\AAPL_5m.csv')
    
    extraer_datos_ibkr_1m(2, IB_conexion, contrato)
    df_1m = pd.read_csv(r'.\AAPL_1m.csv')

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
            * "1 min" (solo deja extraer el último mes)
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
        - Sobrescribir los datos nuevos a los ya existentes, en el caso que ya se tenga el archivo.
    """
