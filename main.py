from fetcher import get_spot, fetch_option
from analisi import netgex, walls
import numpy as np

print("""Software per l'analisi di mercato,
      al momento supporta solo cripto,
      è in grado di rilevare supporti e resistenze dinamiche""")
currency = str(input("Scegli il simbolo da analizzare: ETH, BTC\n"))
spot = get_spot(currency)
print(f"L'attuale prezzo spot è: {spot}")
expiry = str(input("\nInserisci la data dell'opzione da analizzare\n"))
df = fetch_option(expiry, currency)
df, Netgex = netgex(df, spot)
print(f"Netgex = {Netgex}")
df, call_walls, put_walls = walls(df, spot)
print(df)
print(f"""Call walls= {call_walls}""")
print(f"""Put walls= {put_walls}""")