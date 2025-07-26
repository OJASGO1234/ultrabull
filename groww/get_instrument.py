from . import growwapi
from langchain_core.tools import tool

@tool
def get_instrument_by_groww_symbol(groww_symbol: str):
    """Get instrument details by Groww symbol (e.g., 'NSE-RELIANCE').
    The response contains the following fields:

Name	Type	Description
exchange	string	The Exchange where the instrument is traded
exchange_token	string	The unique token assigned to the instrument by the exchange
trading_symbol	string	The trading symbol of the instrument to place orders with
groww_symbol	string	The symbol used by Groww to identify the instrument
name	string	The name of the instrument
instrument_type	string	The type of the instrument
segment	string	Segment of the instrument such as CASH, FNO etc.
series	string	The series of the instrument (e.g., EQ, A, B, etc.)
isin	string	The ISIN (International Securities Identification number) (International Securities Identification Number) of the instrument
underlying_symbol	string	The symbol of the underlying asset (for derivatives). Empty for stocks and indices
underlying_exchange_token	string	The exchange token of the underlying asset
lot_size	number	The minimum lot size for trading the instrument
expiry_date	string	The expiry date of the instrument (for Derivatives)
strike_price	number	The strike price of the instrument (for Options)
tick_size	number	The minimum price movement for the instrument
freeze_quantity	number	The quantity that is frozen for trading
is_reserved	boolean	Whether the instrument is reserved for trading
buy_allowed	boolean	Whether buying the instrument is allowed
sell_allowed	boolean	Whether selling the instrument is allowed

    """
    print("get_instrument_by_groww_symbol called!!")
    resp = growwapi.get_instrument_by_groww_symbol(groww_symbol)
    print(resp)
    return resp